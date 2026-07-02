"""Policy backends: OpenVLA (baseline) and VLA-JEPA (headliner).

Both lazy-import their heavy stacks at construction time. OpenVLA runs in-process via
Hugging Face ``predict_action``; VLA-JEPA talks to the official StarVLA WebSocket policy
server. Keep the two policy stacks in separate environments/images. See NOTES.md.
"""
