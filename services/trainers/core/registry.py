"""Registry of named trainer plugins."""

from __future__ import annotations

from voiceloop_common import VoiceLoopError

from .base import TrainerPlugin


class TrainerRegistryError(VoiceLoopError):
    """Base class for registry lookup and registration failures."""


class PluginNotFoundError(TrainerRegistryError):
    """No plugin is registered under the requested name."""


class DuplicatePluginError(TrainerRegistryError):
    """A plugin is already registered under the requested name."""


class TrainerRegistry:
    """In-memory registry mapping stable plugin names to trainer instances."""

    def __init__(self) -> None:
        """Create an empty registry."""
        self._plugins: dict[str, TrainerPlugin] = {}

    def register(self, plugin: TrainerPlugin, *, replace: bool = False) -> None:
        """Register a plugin under its ``name`` property.

        Args:
            plugin: Trainer plugin instance to register.
            replace: Allow overwriting an existing name (default: reject).

        Raises:
            TrainerRegistryError: If the plugin name is empty.
            DuplicatePluginError: If the name exists and ``replace`` is false.

        """
        name = plugin.name
        if not name:
            raise TrainerRegistryError("trainer plugin name must be a non-empty string")
        if name in self._plugins and not replace:
            raise DuplicatePluginError(f"trainer plugin already registered: {name!r}")
        self._plugins[name] = plugin

    def get(self, name: str) -> TrainerPlugin:
        """Return the plugin registered under ``name``.

        Raises:
            PluginNotFoundError: If no such plugin is registered.

        """
        try:
            return self._plugins[name]
        except KeyError as exc:
            raise PluginNotFoundError(f"no trainer plugin registered as {name!r}") from exc

    def list(self) -> list[str]:
        """Return all registered plugin names in sorted order."""
        return sorted(self._plugins)
