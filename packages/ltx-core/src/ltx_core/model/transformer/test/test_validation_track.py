"""Isolated AST/control-flow regression tests. No Torch, GPU or FSDP runtime."""
import ast
import copy
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class Tensor(np.ndarray):
    def __iter__(self):
        return (np.asarray(value).view(Tensor) for value in super().__iter__())

    @property
    def device(self):
        return 'cpu'

    def to(self, *args, device=None, dtype=None, **kwargs):
        if args and args[0] != 'cpu':
            dtype = args[0]
        return np.asarray(self, dtype=dtype).view(Tensor)

    def unsqueeze(self, dim):
        return np.expand_dims(self, dim)

    def transpose(self, a, b):
        return np.swapaxes(self, a, b)

    def expand(self, *shape):
        shape = tuple(s if s != -1 else self.shape[i] for i, s in enumerate(shape))
        return np.broadcast_to(self, shape).view(Tensor)

    def contiguous(self):
        return self.copy()

    def float(self):
        return self.to(dtype=np.float32)

    def numel(self):
        return self.size


def tensor(value):
    return np.asarray(value, dtype=np.float32).view(Tensor)


def interpolate(x, size, mode, align_corners=None):
    if mode == 'nearest':
        return np.take(x, np.floor(np.arange(size) * x.shape[-1] / size).astype(int), axis=-1)
    return tensor([[np.interp(np.linspace(0, x.shape[-1] - 1, size), np.arange(x.shape[-1]), row)
                    for row in batch] for batch in x])


class Module:
    def __init__(self, **kwargs):
        self.training = True
        self.__dict__.update(kwargs)

    def modules(self):
        pending, visited = [self], set()
        while pending:
            m = pending.pop()
            if id(m) in visited:
                continue
            visited.add(id(m))
            yield m
            for v in m.__dict__.values():
                pending.extend([v] if isinstance(v, Module) else
                               [item for item in v if isinstance(item, Module)] if isinstance(v, list) else [])

    def eval(self):
        for m in self.modules():
            m.training = False


@dataclass(frozen=True)
class Modality:
    latent: Tensor
    sigma: Tensor
    timesteps: Tensor
    positions: Tensor
    context: object
    enabled: bool = True
    context_mask: object = None
    attention_mask: object = None


@dataclass(frozen=True)
class LatentState:
    latent: Tensor
    denoise_mask: Tensor
    positions: Tensor
    clean_latent: Tensor
    attention_mask: object = None


def state(tokens):
    return LatentState(tensor(np.zeros((1, tokens, 2))), tensor(np.ones((1, tokens, 1))),
                       tensor(np.zeros((1, 1, tokens, 2))), tensor(np.zeros((1, tokens, 2))))


torch = NS(nn=NS(Module=Module), Tensor=Tensor, float32=np.float32, bfloat16=np.float32,
           bool=np.bool_, no_grad=lambda: lambda f: f,
           zeros_like=lambda x, **kw: np.zeros_like(x, **kw).view(Tensor),
           zeros=lambda *shape, device=None, dtype=None: np.zeros(shape, dtype=dtype).view(Tensor),
           cat=lambda xs, dim: np.concatenate(xs, axis=dim).view(Tensor),
           where=lambda c, x, y: np.where(c, x, y).view(Tensor),
           Generator=lambda **kw: NS(manual_seed=lambda seed: object()))


class Harness:
    def run(self, **kwargs):
        self.run_kwargs = kwargs
        return self.run_callback(kwargs)


env = dict(torch=torch, Tensor=Tensor, Modality=Modality, LatentState=LatentState,
           F=NS(interpolate=interpolate), dataclass=dataclass, replace=replace,
           Path=Path, Harness=Harness)


def load_classes(filename, classes):
    tree = ast.parse((ROOT / filename).read_text())
    selected = []
    for name, renamed, methods, bases in classes:
        cls = copy.deepcopy(next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name))
        cls.name = renamed
        if methods is not None:
            cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in methods]
        if bases is not None:
            cls.bases = [ast.Name(id=b, ctx=ast.Load()) for b in bases]
        selected.append(cls)
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), *selected], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), filename, 'exec'), env)


load_classes('validation_runner_bakcup.py', [
    ('ValidationModality', 'ValidationModality', None, None),
    ('ValidationRunner', 'BaseRunner', {'_modality_from_latent_state', '_generate_sample', '_apply_video_conditionings'}, ['Harness']),
])
load_classes('validation_runner.py', [
    ('CachedTrack', 'CachedTrack', None, None), ('ActiveTrack', 'ActiveTrack', None, None),
    ('ValidationRunner', 'Runner', {'_model_candidates', '_uses_track_conditioning', '_validate_conditioning',
                                 'run', '_align_track_to_audio', '_run_denoising', '_generate_sample'}, ['BaseRunner']),
])
Runner, BaseRunner = env['Runner'], env['BaseRunner']


class Guider:
    def __init__(self, scale):
        self.scale = scale

    def enabled(self):
        return self.scale != 0

    def delta(self, positive, negative):
        return (positive - negative) * self.scale


class X0:
    def __init__(self, wrapped):
        self.wrapped = wrapped

    def __call__(self, video, audio, perturbations):
        self.wrapped.calls.append((video, audio, perturbations))
        return (video.latent if video is not None else None,
                audio.latent if audio is not None else None)


env.update(X0Model=X0, CFGGuider=Guider, STGGuider=Guider,
           LTX2Scheduler=lambda: NS(execute=lambda steps: tensor([1.0, 0.5, 0.0])),
           EulerDiffusionStep=lambda: NS(step=lambda latent, denoised, sigmas, index: denoised),
           GaussianNoiser=lambda **kw: lambda s, **kw: s)


def split_counts(modality):
    tree = ast.parse((ROOT / 'transformer_args.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MultiModalTransformerArgsPreprocessor')
    prepare = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'prepare')
    branch = next(n for n in prepare.body if isinstance(n, ast.If) and ast.unparse(n.test) == 'self.is_count_video_ref')
    statements = [copy.deepcopy(n) for n in branch.body if not isinstance(n, ast.Return)]
    scope = dict(modality=modality)
    exec(compile(ast.Module(body=statements, type_ignores=[]), 'actual-split-logic', 'exec'), scope)
    return scope['target_token_count'], scope['reference_token_count']


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.runner = Runner()
        self.sample = NS(conditions=[NS(type='reference', video='reference.mp4')], video_dims=None, seed=None)
        self.runner._config = NS(samples=[self.sample], generate_audio=True, generate_video=True,
                                 inference_steps=2, guidance_scale=2, stg_scale=1,
                                 stg_blocks=[0], stg_mode='stg_av', video_dims=(64, 64, 121),
                                 frame_rate=24, seed=42)
        self.cached = env['CachedTrack'](tensor([[0, 0], [1, 1]]), np.asarray([True, True]).view(Tensor), Path('track.pt'))
        self.runner._cached_tracks = {id(self.sample): self.cached}
        self.runner._active_track = None
        self.runner._build_stg_perturbation_config = lambda *args: 'STG'
        self.runner._post_process_latent = lambda denoised, *args: denoised

    def model(self, mode='v2', rope=True, timesteps=True):
        param = NS(requires_grad=True)
        block = Module(use_simple_summary=True, simple_summary=Module(), simple_summary_scale=param)
        if mode == 'v1':
            block.spatial_track_encoder_version1 = Module()
        elif mode == 'v2':
            block.scale_shift_table_audio_track = param
        return Module(track_injection=mode, use_track_rope=rope,
                      residual_track_rope=Module() if rope else None,
                      spatial_track_encoder=Module() if mode == 'v2' else None,
                      apply_track_with_timesteps=timesteps,
                      track_timesteps_adaln=Module() if timesteps else None,
                      transformer_blocks=[block], _track_conditioning_modules_initialized=True,
                      calls=[], parameter=param)

    def test_v1_and_v2_detected_without_rope(self):
        for mode in ('v1', 'v2'):
            self.assertTrue(Runner._uses_track_conditioning(self.model(mode, False, False)))

    def test_rope_only_and_disabled(self):
        self.assertTrue(Runner._uses_track_conditioning(Module(use_track_rope=True)))
        self.assertFalse(Runner._uses_track_conditioning(Module(transformer_blocks=[])))

    def test_nested_wrappers_do_not_stop_at_empty_outer_blocks(self):
        model = self.model()
        wrapper = Module(transformer_blocks=[], _orig_mod=Module(_fsdp_wrapped_module=Module(base_model=Module(model=model))))
        wrapper.module = wrapper
        self.assertTrue(Runner._uses_track_conditioning(wrapper))
        self.runner._validate_conditioning(wrapper)

    def test_missing_track_checked_for_all_samples_not_local_work(self):
        self.runner._config.samples.append(NS(conditions=[]))
        with self.assertRaisesRegex(ValueError, 'sample 2 requires a track'):
            self.runner.run(self.model(), 0, ROOT, 'cpu', None, work_items=[(0, True)])

    def test_video_only_does_not_require_track(self):
        self.runner._config.generate_audio = False
        self.runner._cached_tracks = {}
        self.runner._validate_conditioning(self.model())

    def test_uninitialized_model_modules(self):
        for attribute in ('spatial_track_encoder', 'residual_track_rope', 'track_timesteps_adaln'):
            with self.subTest(attribute=attribute):
                m = self.model()
                setattr(m, attribute, None)
                with self.assertRaises(RuntimeError):
                    self.runner._validate_conditioning(m)

    def test_uninitialized_block_and_summary(self):
        for mode, attribute in [('v1', 'spatial_track_encoder_version1'), ('v2', 'scale_shift_table_audio_track'), ('v2', 'simple_summary')]:
            with self.subTest(attribute=attribute):
                m = self.model(mode)
                setattr(m.transformer_blocks[0], attribute, None)
                with self.assertRaises(RuntimeError):
                    self.runner._validate_conditioning(m)

    def test_summary_requires_real_reference(self):
        self.sample.conditions = []
        with self.assertRaisesRegex(ValueError, 'reference-video'):
            self.runner._validate_conditioning(self.model())

    def test_frozen_video_with_reference_allowed(self):
        self.sample.conditions.append(NS(type='video_to_audio'))
        self.runner._config.generate_video = False
        self.runner._validate_conditioning(self.model())

    def test_frozen_reference_helper_skips_other_conditions(self):
        applied = []
        class ReferenceCondition:
            def __init__(self, **kwargs):
                applied.append(kwargs)

            def apply_to(self, s, tools):
                return state(s.latent.shape[1] + 2)
        env['VideoConditionByReferenceLatent'] = ReferenceCondition
        reference = NS(type='reference', video='ref.mp4', downscale_factor=2,
                       temporal_scale_factor=1)
        sample = NS(conditions=[NS(type='prefix', video='prefix.mp4'), reference])
        media = NS(conditions={0: NS(latent=tensor([1])), 1: NS(latent=tensor([2]))})
        result = BaseRunner._apply_video_conditionings(state(4), None, sample, media,
                                                       'cpu', references_only=True)
        self.assertEqual(result.latent.shape[1], 6)
        self.assertEqual(len(applied), 1)

    def test_eval_restores_mixed_modes_and_keeps_live_model_and_gradients(self):
        model = self.model()
        model.spatial_track_encoder.training = False
        wrapper = Module(module=model)
        before = [(m, m.training) for m in wrapper.modules()]
        work = [(0, False)]
        def callback(kwargs):
            self.assertIs(kwargs['transformer'], wrapper)
            self.assertIs(kwargs['work_items'], work)
            self.assertTrue(all(not m.training for m in wrapper.modules()))
            self.assertTrue(model.parameter.requires_grad)
            return ['ok']
        self.runner.run_callback = callback
        self.assertEqual(self.runner.run(wrapper, 3, ROOT, 'cpu', None, work_items=work), ['ok'])
        self.assertTrue(all(m.training == training for m, training in before))
        self.assertTrue(model.parameter.requires_grad)

    def test_eval_restored_after_exception(self):
        model = self.model()
        def fail(kwargs):
            raise RuntimeError('generation failed')
        self.runner.run_callback = fail
        with self.assertRaisesRegex(RuntimeError, 'generation failed'):
            self.runner.run(model, 0, ROOT, 'cpu', None)
        self.assertTrue(model.training)
        self.assertIsNone(self.runner._active_track)

    def test_track_alignment_and_reference_suffix(self):
        xy, valid = Runner._align_track_to_audio(cached_track=self.cached, target_audio_tokens=3,
                                                 final_audio_tokens=5, batch_size=2, device='cpu', dtype=np.float32)
        np.testing.assert_allclose(xy[0], [[0, 0], [.5, .5], [1, 1], [0, 0], [0, 0]])
        np.testing.assert_array_equal(valid[0], [1, 1, 1, 0, 0])
        self.assertEqual(xy.shape, (2, 5, 2))

    def test_alignment_rejects_short_final_sequence(self):
        with self.assertRaises(ValueError):
            Runner._align_track_to_audio(cached_track=self.cached, target_audio_tokens=3,
                                         final_audio_tokens=2, batch_size=1, device='cpu', dtype=np.float32)

    def test_modality_preserves_metadata_attention_mask_and_cfg_replace(self):
        s = replace(state(7), attention_mask=object())
        modality = Runner._modality_from_latent_state(s, 'positive', tensor([.5]), target_token_count=4)
        negative = replace(modality, context='negative')
        self.assertEqual((negative.target_token_count, negative.reference_token_count), (4, 3))
        self.assertIs(negative.attention_mask, s.attention_mask)
        self.assertEqual(split_counts(negative), (4, 3))

    def test_preprocessor_preserves_training_fallback(self):
        self.assertEqual(split_counts(NS(latent=state(10).latent)), (8, 2))

    def test_invalid_explicit_split_rejected(self):
        with self.assertRaises(ValueError):
            split_counts(NS(latent=state(7).latent, target_token_count=4, reference_token_count=2))

    def test_modality_rejects_invalid_track_shapes(self):
        with self.assertRaises(ValueError):
            Runner._modality_from_latent_state(state(3), 'ctx', tensor([1]),
                                               track_xy=tensor(np.zeros((1, 2, 2))))
        with self.assertRaises(ValueError):
            Runner._modality_from_latent_state(state(3), 'ctx', tensor([1]),
                                               track_valid=tensor(np.zeros((1, 2))))

    def test_trainer_passes_live_wrapped_model(self):
        tree = ast.parse((ROOT / 'trainer.py').read_text())
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                      and n.name == '_run_validation')
        call = next(n for n in ast.walk(method) if isinstance(n, ast.Call)
                    and ast.unparse(n.func) == 'self._validation_runner.run')
        transformer = next(k.value for k in call.keywords if k.arg == 'transformer')
        self.assertEqual(ast.unparse(transformer), 'self._transformer')

    def test_positive_cfg_stg_all_use_same_tracks_live_wrapper_and_video_split(self):
        model = Module(calls=[])
        self.runner._active_track = env['ActiveTrack'](self.cached)
        steps = []
        self.runner._run_denoising(model, state(7), state(5), None, None,
            video_frozen=False, audio_frozen=False, v_ctx_pos='v+', a_ctx_pos='a+',
            v_ctx_neg='v-', a_ctx_neg='a-', device='cpu', sampling_ctx=NS(advance_step=lambda: steps.append(1)),
            video_target_tokens=4, audio_target_tokens=3)
        self.assertEqual(len(model.calls), 6)
        self.assertEqual(len(steps), 2)
        first_track = model.calls[0][1].track_xy
        for video, audio, perturbation in model.calls:
            self.assertIs(audio.track_xy, first_track)
            self.assertEqual(audio.track_xy.shape, (1, 5, 2))
            self.assertEqual((video.target_token_count, video.reference_token_count), (4, 3))
            np.testing.assert_array_equal(audio.track_valid[0], [1, 1, 1, 0, 0])
        self.assertEqual(model.calls[1][1].context, 'a-')
        self.assertEqual(model.calls[2][2], 'STG')

    def test_audio_only_cfg_executes(self):
        model = Module(calls=[])
        self.runner._config.stg_scale = 0
        self.runner._active_track = env['ActiveTrack'](self.cached)
        self.runner._run_denoising(model, None, state(4), None, None,
            video_frozen=False, audio_frozen=False, v_ctx_pos=None, a_ctx_pos='a+',
            v_ctx_neg=None, a_ctx_neg='a-', device='cpu', sampling_ctx=NS(advance_step=lambda: None),
            audio_target_tokens=4)
        self.assertEqual(len(model.calls), 4)
        self.assertEqual(model.calls[1][1].context, 'a-')

    def test_frozen_audio_keeps_zero_timestep_and_uses_actual_length(self):
        model = Module(calls=[])
        self.runner._active_track = env['ActiveTrack'](self.cached)
        self.runner._run_denoising(model, state(7), state(9), None, None,
            video_frozen=False, audio_frozen=True, v_ctx_pos='v+', a_ctx_pos='a+',
            v_ctx_neg='v-', a_ctx_neg='a-', device='cpu', sampling_ctx=NS(advance_step=lambda: None),
            video_target_tokens=4, audio_target_tokens=9)
        for video, audio, perturbation in model.calls:
            self.assertEqual(audio.track_xy.shape, (1, 9, 2))
            np.testing.assert_array_equal(audio.timesteps, 0)

    def test_generation_never_calls_model_level_ste_and_counts_before_references(self):
        r = self.runner
        r._get_prompt_embeddings = lambda *args: ('v+', 'a+', 'v-', 'a-')
        r._create_video_tools = lambda *args: NS(create_initial_state=lambda **kw: state(4))
        r._create_audio_tools = lambda *args: NS(create_initial_state=lambda **kw: state(3))
        r._apply_video_conditionings = lambda s, *args, **kw: state(s.latent.shape[1] + 2)
        r._apply_audio_conditionings = lambda s, *args: state(s.latent.shape[1] + 1)
        recorded = []
        r._run_denoising = lambda **kw: (recorded.append(kw) or (kw['video_state'], kw['audio_state']))
        r._decode_video = r._decode_audio = lambda *args: None
        r._finalize_modality = lambda s, *args: None
        r._find_condition_media = lambda *args: NS(latent=tensor([1]))
        model = self.model()
        model.spatial_track_encoder = lambda *args: self.fail('STE called outside root forward')
        for frozen in (False, True):
            if frozen:
                self.sample.conditions.append(NS(type='video_to_audio'))
            r._generate_sample(self.sample, None, None, model, 'cpu', None)
            result = recorded[-1]
            self.assertIs(result['transformer'], model)
            self.assertEqual((result['video_target_tokens'], result['audio_target_tokens']), (4, 3))
            self.assertEqual((result['video_state'].latent.shape[1], result['audio_state'].latent.shape[1]), (6, 4))
            self.assertIsNone(r._active_track)


if __name__ == '__main__':
    unittest.main(verbosity=2)