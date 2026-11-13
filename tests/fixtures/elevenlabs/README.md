Real error bodies from the ElevenLabs API, recorded 2026-11-13 on Windows
(tests/live/test_elevenlabs_live.py and a `GET /v1/user/subscription` call).
Tests classify these exact bytes, so a format change shows up as a failing test.

Note: real answers carry both the current `code` and the legacy `status`;
`status` is the more specific one (`unauthorized` vs `missing_permissions`).

- `401_invalid_api_key.json`: text-to-speech with a made-up key.
- `404_voice_not_found.json`: text-to-speech with voice_id `ThisVoiceDoesNotExist`.
- `401_missing_permissions.json`: a key without the `user_read` permission.

The success answer had the header `character-cost: 17` for 43 characters on
eleven_multilingual_v2 (no `x-character-count`), and the account was charged 17.
