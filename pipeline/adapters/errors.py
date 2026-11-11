"""Errors adapters raise so the runner can decide what to do."""


class TransientAdapterError(Exception):
    """A provider hiccup worth retrying: timeout, rate limit, 5xx, dropped connection.

    Real adapters translate their SDK's errors into this; the chaos fakes raise it.
    """
