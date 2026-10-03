# Bow models

3D bow models are **not committed** — `.gitignore` covers `*.obj`, `*.mtl`,
`*.glb`, `*.gltf` and `*.fbx` in this directory. They are third-party or
hand-made, they are not part of the source, and the game does not need one:
`render/bow_model.py` falls back to the procedural bow when no asset is
present, and the tests that read one skip.

Drop a model here and it is picked up automatically. The path is
`config.BOW_ASSET_PATH`. Preview one without starting the game:

    python -m fletchflow.render.bow_model assets/models/<file>.obj --yaw 90 --out preview.png

Format and authoring requirements — `.obj` or `.glb`, bow and arrow as
separate objects, the naming rules the loader uses — are in PLAN.md §4.7.8a.

## Currently in use (local only)

`FletchFlow_BowArrow.obj` — the custom FletchFlow bow and arrow, made in
Blender for this project (no third-party content, so no credit is owed).
Objects `Bow` (380 tris, low-poly recurve) and `Arrow` (114 tris, three vanes),
no string: the game draws its own. Its source files — the rigged `.blend`s,
`.glb`s and build scripts — live outside this repo in the Blender workspace
under `Blender Models/FletchFlow/`; re-export there and copy the `.obj` here.

It replaced `ANIMATEDBOW.obj` (a free Sketchfab placeholder) on 2026-10-03.

If a third-party model is ever shipped with the game, record its title, author,
source URL and licence here first. Most free Sketchfab models are CC-BY, which
permits redistribution but **requires** credit.

## hand_landmarker.task

MediaPipe hand landmarker, downloaded rather than committed (see README).
Google, Apache-2.0.
