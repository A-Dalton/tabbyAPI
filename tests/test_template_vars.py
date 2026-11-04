import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException

from common import model
from common.templating import PromptTemplate
from endpoints.OAI.types.chat_completion import ChatCompletionRequest
from endpoints.OAI.utils.chat_completion import (
    PRESERVE_REASONING_VARS,
    _resolve_start_in_reasoning,
    apply_chat_template,
    resolve_template_vars,
)


def request(**kwargs):
    return ChatCompletionRequest(messages=[{"role": "user", "content": "hi"}], **kwargs)


def request_with_reasoning(reasoning="thoughts", **kwargs):
    return ChatCompletionRequest(
        messages=[
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello", "reasoning": reasoning},
            {"role": "user", "content": "more"},
        ],
        **kwargs,
    )


def container(default=None, force=None):
    return SimpleNamespace(
        template_vars_default=default or {},
        template_vars_force=force or {},
    )


class ResolveTemplateVarsTests(unittest.TestCase):
    def test_defaults_applied(self):
        out = resolve_template_vars(request(), container(default={"reasoning_effort": "high"}))
        self.assertEqual(out["reasoning_effort"], "high")

    def test_client_template_vars_override_defaults(self):
        out = resolve_template_vars(
            request(template_vars={"reasoning_effort": "no_think"}),
            container(default={"reasoning_effort": "high"}),
        )
        self.assertEqual(out["reasoning_effort"], "no_think")

    def test_top_level_reasoning_effort_mapped(self):
        out = resolve_template_vars(request(reasoning_effort="low"), container())
        self.assertEqual(out["reasoning_effort"], "low")

    def test_top_level_reasoning_effort_overrides_defaults(self):
        out = resolve_template_vars(
            request(reasoning_effort="low"),
            container(default={"reasoning_effort": "high"}),
        )
        self.assertEqual(out["reasoning_effort"], "low")

    def test_explicit_template_var_beats_top_level(self):
        out = resolve_template_vars(
            request(
                reasoning_effort="low",
                template_vars={"reasoning_effort": "high"},
            ),
            container(),
        )
        self.assertEqual(out["reasoning_effort"], "high")

    def test_force_overrides_client(self):
        out = resolve_template_vars(
            request(
                reasoning_effort="no_think",
                template_vars={"enable_thinking": False},
            ),
            container(force={"reasoning_effort": "high", "enable_thinking": True}),
        )
        self.assertEqual(out["reasoning_effort"], "high")
        self.assertTrue(out["enable_thinking"])

    def test_chat_template_kwargs_alias(self):
        out = resolve_template_vars(
            request(chat_template_kwargs={"reasoning_effort": "high"}), container()
        )
        self.assertEqual(out["reasoning_effort"], "high")

    def test_unrelated_vars_pass_through(self):
        out = resolve_template_vars(
            request(template_vars={"custom": 1}),
            container(default={"other": 2}, force={"third": 3}),
        )
        self.assertEqual(out, {"custom": 1, "other": 2, "third": 3})

    def test_enable_thinking_mapped(self):
        out = resolve_template_vars(request(enable_thinking=False), container())
        self.assertIs(out["enable_thinking"], False)

    def test_verbosity_mapped(self):
        out = resolve_template_vars(request(verbosity="low"), container())
        self.assertEqual(out["verbosity"], "low")

    def test_reasoning_object_mapped(self):
        out = resolve_template_vars(
            request(reasoning={"effort": "high", "enabled": True}), container()
        )
        self.assertEqual(out["reasoning_effort"], "high")
        self.assertIs(out["enable_thinking"], True)

    def test_flat_fields_beat_reasoning_object(self):
        out = resolve_template_vars(
            request(
                reasoning={"effort": "high", "enabled": True},
                reasoning_effort="no_think",
                enable_thinking=False,
            ),
            container(),
        )
        self.assertEqual(out["reasoning_effort"], "no_think")
        self.assertIs(out["enable_thinking"], False)

    def test_reasoning_object_max_tokens_ignored(self):
        out = resolve_template_vars(request(reasoning={"max_tokens": 4096}), container())
        self.assertNotIn("reasoning_effort", out)
        self.assertNotIn("enable_thinking", out)
        self.assertNotIn("max_tokens", out)

    def test_reasoning_object_unknown_keys_ignored(self):
        out = resolve_template_vars(
            request(reasoning={"effort": "low", "exclude": True}), container()
        )
        self.assertEqual(out["reasoning_effort"], "low")
        self.assertNotIn("exclude", out)

    def test_template_vars_beat_flat_fields(self):
        out = resolve_template_vars(
            request(
                enable_thinking=False,
                verbosity="low",
                template_vars={"enable_thinking": True, "verbosity": "high"},
            ),
            container(),
        )
        self.assertIs(out["enable_thinking"], True)
        self.assertEqual(out["verbosity"], "high")

    def test_reasoning_in_the_request_is_preserved(self):
        out = resolve_template_vars(request_with_reasoning(), container())
        self.assertEqual(out, PRESERVE_REASONING_VARS)

    def test_nothing_to_preserve_changes_nothing(self):
        self.assertEqual(resolve_template_vars(request(), container()), {})

    def test_empty_reasoning_is_nothing_to_preserve(self):
        self.assertEqual(resolve_template_vars(request_with_reasoning(""), container()), {})

    def test_model_default_overrides_preserving(self):
        out = resolve_template_vars(
            request_with_reasoning(), container(default={"preserve_thinking": False})
        )
        self.assertIs(out["preserve_thinking"], False)
        self.assertIs(out["clear_thinking"], False)

    def test_request_overrides_preserving(self):
        out = resolve_template_vars(
            request_with_reasoning(template_vars={"clear_thinking": True}), container()
        )
        self.assertIs(out["clear_thinking"], True)

    def test_forced_value_overrides_preserving(self):
        out = resolve_template_vars(
            request_with_reasoning(template_vars={"preserve_thinking": True}),
            container(force={"preserve_thinking": False}),
        )
        self.assertIs(out["preserve_thinking"], False)


class ReasoningOffTests(unittest.TestCase):
    """reasoning_effort "none" as enable_thinking: false, for templates rejecting "none"."""

    def resolve(self, data, mc=None):
        return resolve_template_vars(data, mc or container(), reasoning_off=True)

    def test_none_turns_thinking_off(self):
        self.assertEqual(self.resolve(request(reasoning_effort="none")), {"enable_thinking": False})

    def test_none_from_any_source(self):
        for data, mc in (
            (request(reasoning={"effort": "none"}), None),
            (request(template_vars={"reasoning_effort": "none"}), None),
            (request(), container(default={"reasoning_effort": "none"})),
            (request(), container(force={"reasoning_effort": "none"})),
        ):
            self.assertEqual(self.resolve(data, mc), {"enable_thinking": False})

    def test_other_efforts_unchanged(self):
        for effort in ("low", "high", "no_think"):
            out = self.resolve(request(reasoning_effort=effort))
            self.assertEqual(out, {"reasoning_effort": effort})

    def test_only_with_reasoning_off(self):
        out = resolve_template_vars(request(reasoning_effort="none"), container())
        self.assertEqual(out, {"reasoning_effort": "none"})

    def test_forced_effort_wins_over_none(self):
        out = self.resolve(
            request(reasoning_effort="none"), container(force={"reasoning_effort": "high"})
        )
        self.assertEqual(out, {"reasoning_effort": "high"})

    def test_none_wins_over_default_enable_thinking(self):
        out = self.resolve(
            request(reasoning_effort="none"),
            container(default={"enable_thinking": True, "reasoning_effort": "high"}),
        )
        self.assertEqual(out, {"enable_thinking": False})

    def test_enable_thinking_from_the_request_kept(self):
        for data in (
            request(reasoning_effort="none", enable_thinking=True),
            request(reasoning={"effort": "none", "enabled": True}),
            request(reasoning_effort="none", template_vars={"enable_thinking": True}),
        ):
            self.assertEqual(self.resolve(data), {"enable_thinking": True})

    def test_forced_enable_thinking_kept(self):
        out = self.resolve(
            request(reasoning_effort="none"), container(force={"enable_thinking": True})
        )
        self.assertEqual(out, {"enable_thinking": True})


# Rejects reasoning_effort "none" and turns reasoning off with enable_thinking,
# as DeepSeek V4's does: <think> opens the generation prompt, </think> skips it
REJECTS_NONE = (
    "{% if reasoning_effort is defined %}"
    "{% if reasoning_effort not in ['low', 'high', 'max'] %}"
    "{{ raise_exception('Invalid reasoning effort: ' ~ reasoning_effort) }}"
    "{% endif %}[effort {{ reasoning_effort }}]{% endif %}"
    "{% for m in messages %}<|{{ m.role }}|>{{ m.content }}{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>"
    "{% if enable_thinking is defined and not enable_thinking %}</think>"
    "{% else %}<think>{% endif %}{% endif %}"
)
# Takes reasoning_effort as it is, as gpt-oss's does
ACCEPTS_NONE = (
    "{% if reasoning_effort is defined %}Reasoning: {{ reasoning_effort }}{% endif %}"
    "{% if enable_thinking is defined %}[thinking {{ enable_thinking }}]{% endif %}"
    "{% for m in messages %}<|{{ m.role }}|>{{ m.content }}{% endfor %}"
)
ALWAYS_RAISES = "{{ raise_exception('Broken template') }}"


class RejectedNoneRenderTests(unittest.IsolatedAsyncioTestCase):
    async def render(self, template, data):
        """The prompt, and whether start_in_reasoning: auto starts in reasoning."""

        mc = SimpleNamespace(
            prompt_template=PromptTemplate("test", template),
            use_vision=False,
            harmony=False,
            muse_glimmer=False,
            reasoning=True,
            reasoning_start_token="<think>",
            reasoning_end_token="</think>",
            start_in_reasoning="auto",
            template_vars_default={},
            template_vars_force={},
            get_special_tokens=lambda: {},
            hf_model=SimpleNamespace(add_bos_token=lambda: False),
        )
        with patch.object(model, "container", mc):
            prompt, _ = await apply_chat_template(data)
            return prompt, _resolve_start_in_reasoning(prompt, data)

    async def test_none_renders_without_reasoning(self):
        data = request(reasoning_effort="none")
        prompt, starts_in_reasoning = await self.render(REJECTS_NONE, data)
        self.assertEqual(prompt, "<|user|>hi<|assistant|></think>")
        self.assertFalse(starts_in_reasoning)
        self.assertIs(data.template_vars["enable_thinking"], False)
        self.assertNotIn("reasoning_effort", data.template_vars)

    async def test_other_efforts_render_as_before(self):
        prompt, starts_in_reasoning = await self.render(
            REJECTS_NONE, request(reasoning_effort="high")
        )
        self.assertEqual(prompt, "[effort high]<|user|>hi<|assistant|><think>")
        self.assertTrue(starts_in_reasoning)

    async def test_other_rejected_efforts_still_fail(self):
        with self.assertRaises(HTTPException) as raised:
            await self.render(REJECTS_NONE, request(reasoning_effort="minimal"))
        self.assertEqual(raised.exception.status_code, 400)

    async def test_explicit_enable_thinking_kept(self):
        prompt, starts_in_reasoning = await self.render(
            REJECTS_NONE, request(reasoning_effort="none", enable_thinking=True)
        )
        self.assertEqual(prompt, "<|user|>hi<|assistant|><think>")
        self.assertTrue(starts_in_reasoning)

    async def test_template_accepting_none_unchanged(self):
        data = request(reasoning_effort="none")
        prompt, _ = await self.render(ACCEPTS_NONE, data)
        self.assertEqual(prompt, "Reasoning: none<|user|>hi")

    async def test_failing_fallback_still_fails(self):
        with self.assertRaises(HTTPException) as raised:
            await self.render(ALWAYS_RAISES, request(reasoning_effort="none"))
        self.assertEqual(raised.exception.status_code, 400)


if __name__ == "__main__":
    unittest.main()
