"""
motor.py — the motor system: the brain's output byte-stream becomes ACTION.

Symmetric to senses.py. Sensing turns the world into a byte-stream the brain reads;
the motor system turns the brain's emitted byte-stream into things it DOES:

  write / speak  -> the brain generates a byte-stream, decoded to text = an utterance
                    (the motor act of writing; "speaking" is the same act, a different
                    effector). The utterance is what it says to Claude / types.
  navigate       -> an emitted command selects what to look at next (a URL / scroll).

Honest scope: the byte-brain is small, so early utterances are babble. That is on
purpose and useful — like an infant babbling: the brain SPEAKS (motor), a fluent
teacher (Claude Sonnet 5) hears the babble and replies with correct, related words,
and the brain LEARNS from that reply (sensory). Motor output drives the loop even
before it is fluent; fluency is what the loop is for.
"""
from __future__ import annotations


def utter(brain, seed="", n=120, temperature=0.4):
    """The motor act of writing/speaking: the brain emits a byte-stream -> text."""
    return brain.generate(seed, n=n, temperature=temperature).strip()


def teach_prompt(utterance, topic=None):
    """Frame the brain's (often babbled) utterance so a fluent teacher replies with a
    short, correct lesson the brain can learn from — a parent answering a baby's babble."""
    u = (utterance or "").replace("\n", " ")[:200]
    if topic:
        return (f"You are teaching a small learning 'brain'. It is curious about: {topic}. "
                f"It just babbled: \"{u}\". Reply with 4-8 sentences of clear, correct, simple "
                f"plain prose teaching it something true and concrete about that topic. "
                f"No lists, no markdown — just teach.")
    return (f"You are teaching a small learning 'brain' that is just starting to form words. "
            f"It babbled: \"{u}\". Reply with 4-8 sentences of clear, correct, simple plain "
            f"prose about an everyday true thing (nature, science, daily life), so it can learn "
            f"real words and facts. No lists, no markdown — just teach.")


def choose_curiosity(brain, topics, step):
    """A tiny 'what shall I attend to next' policy. The brain's own utterance biases the
    choice (curiosity), but the loop keeps a topic curriculum so learning stays grounded."""
    if not topics:
        return None
    return topics[step % len(topics)]
