"""Errors adapters raise so the runner can decide what to do."""


class TransientAdapterError(Exception):
    """A provider hiccup worth retrying: timeout, rate limit, 5xx, dropped connection.

    Real adapters translate their SDK's errors into this; the chaos fakes raise it.
    """


class PermanentAdapterError(Exception):
    """Retrying will not help until a human acts: not logged in, usage limit reached,
    unknown model, CLI not installed. The runner stops at once and alerts with this text.
    """
