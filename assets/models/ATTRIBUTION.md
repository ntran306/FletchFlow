# Third-party assets

## ANIMATEDBOW.obj

Downloaded from Sketchfab as a free model (originally `ANIMATEDBOW.fbx`,
converted to OBJ in Blender — Python has no pip-installable FBX loader, see
PLAN.md §4.7.8a). The `.mtl` and its textures were not part of the download
and are not needed: the renderer lights the mesh itself.

**TODO — fill in before any public release.** Most free Sketchfab models are
CC-BY, which permits use and redistribution but *requires* credit. Record here:

- Model title:
- Author:
- Sketchfab URL:
- Licence (e.g. CC-BY 4.0, CC0):

If the licence turns out to forbid redistribution, delete the file, add
`assets/models/*.obj` to `.gitignore`, and replace it with a self-made model —
the game falls back to procedural geometry when no asset is present, so
nothing breaks in the meantime.

## hand_landmarker.task

MediaPipe hand landmarker, downloaded rather than committed (see README).
Google, Apache-2.0.
