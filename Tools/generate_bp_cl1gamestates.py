"""
Generate BP_CL1GameStates - a Pawn Blueprint that owns a CL1 Game States
component (UCl1GameStates), the encoder/decoder that maps four game states to
CL1 stimulation and back.

WHAT IT MAKES
-------------
A Pawn Blueprint at /Game/Blueprints/BP_CL1GameStates with:
  * a `CL1GameStates` component added to its construction script, and
  * sensible defaults set on that component (auto-decode on, a confidence gate).

The component already does the encode/decode work in C++, so the only graph
logic you add by hand is the two-line setup + the movement hook (printed at the
end of this script, and documented in the plugin README).

HOW TO RUN
----------
Requires the Python Editor Script Plugin enabled (Edit > Plugins > "Python
Editor Script Plugin"), and the UeCl1Api plugin compiled.

  * In-editor: Tools > Execute Python Script... and pick this file, or in the
    Output Log's Cmd (set to "Python"):  exec(open(r"<path>/generate_bp_cl1gamestates.py").read())
  * Headless / on launch:
      "/Users/Shared/Epic Games/UE_5.8/Engine/Binaries/Mac/UnrealEditor" \
        "/Users/jenn/Documents/UnrealProjects/CallosumUE/Callosum.uproject" \
        -run=pythonscript -script="<path>/generate_bp_cl1gamestates.py"
"""
import unreal

ASSET_NAME   = "BP_CL1GameStates"
PACKAGE_PATH = "/Game/Blueprints"
ASSET_PATH   = f"{PACKAGE_PATH}/{ASSET_NAME}"
PARENT_CLASS = unreal.Pawn          # so it can possess/drive a game pawn
COMPONENT_CLASS = unreal.Cl1GameStates
COMPONENT_NAME  = "CL1GameStates"

# Defaults pushed onto the component template (snake_case = UE Python naming).
COMPONENT_DEFAULTS = {
    "auto_decode": True,             # bAutoDecode: tick-decode + broadcast
    "decode_interval_seconds": 0.25,
    "decode_window_seconds": 0.5,
    "min_confidence": 0.35,          # only fire OnStateDecoded when fairly sure
    "class_electrodes": [10, 26, 42, 50],
}


def _log(msg):
    unreal.log(f"[BP_CL1GameStates] {msg}")


def create_blueprint():
    asset_tools = unreal.AssetToolsHelpers.get_asset_tools()

    if unreal.EditorAssetLibrary.does_asset_exist(ASSET_PATH):
        _log(f"{ASSET_PATH} exists - deleting to regenerate")
        unreal.EditorAssetLibrary.delete_asset(ASSET_PATH)

    factory = unreal.BlueprintFactory()
    factory.set_editor_property("parent_class", PARENT_CLASS)
    bp = asset_tools.create_asset(ASSET_NAME, PACKAGE_PATH, None, factory)
    if not bp:
        raise RuntimeError("create_asset returned None (check the path is writable)")
    _log(f"created {ASSET_PATH} (parent {PARENT_CLASS.__name__})")
    return bp


def add_component(bp):
    """Add the CL1GameStates component to the Blueprint via SubobjectDataSubsystem
    (the UE5 way to script components onto a Blueprint's construction script)."""
    subsys = unreal.get_engine_subsystem(unreal.SubobjectDataSubsystem)
    handles = subsys.k2_gather_subobject_data_for_blueprint(bp)
    if not handles:
        raise RuntimeError("no root subobject handle for the Blueprint")
    root = handles[0]

    params = unreal.AddNewSubobjectParams(
        parent_handle=root, new_class=COMPONENT_CLASS, blueprint_context=bp)
    new_handle, fail = subsys.add_new_subobject(params)
    if not fail.is_empty():
        raise RuntimeError(f"add_new_subobject failed: {fail}")
    try:
        subsys.rename_subobject(new_handle, COMPONENT_NAME)
    except Exception as e:
        _log(f"WARNING: rename to '{COMPONENT_NAME}' failed ({e}); keeping default name")
    _log(f"added component '{COMPONENT_NAME}' ({COMPONENT_CLASS.__name__})")
    return new_handle


def apply_defaults(bp, handle):
    data = unreal.SubobjectDataBlueprintFunctionLibrary.get_data(handle)
    template = unreal.SubobjectDataBlueprintFunctionLibrary.get_object_for_blueprint(data, bp)
    if not template:
        _log("WARNING: could not resolve component template; skipping defaults")
        return
    for prop, value in COMPONENT_DEFAULTS.items():
        try:
            template.set_editor_property(prop, value)
        except Exception as e:
            _log(f"WARNING: could not set '{prop}' = {value} ({e})")
    _log("applied component defaults")


def compile_blueprint(bp):
    """Compile the Blueprint. UE 5.x exposes this via BlueprintEditorLibrary;
    older builds used KismetEditorUtilities - try both."""
    if hasattr(unreal, "BlueprintEditorLibrary"):
        unreal.BlueprintEditorLibrary.compile_blueprint(bp)
    elif hasattr(unreal, "KismetEditorUtilities"):
        unreal.KismetEditorUtilities.compile_blueprint(bp)
    else:
        _log("WARNING: no compile API found; the asset will compile on open")


def main():
    bp = create_blueprint()
    handle = add_component(bp)
    apply_defaults(bp, handle)

    compile_blueprint(bp)
    unreal.EditorAssetLibrary.save_asset(ASSET_PATH)
    _log(f"compiled + saved {ASSET_PATH}")

    _log("NEXT (event graph, add in-editor):")
    _log("  1. BeginPlay -> CL1 Bridge (GetCl1Bridge) -> StartReceiver(12345)")
    _log("       + ConfigureControlTarget(bridge host, 12346)")
    _log("  2. Bind 'On State Decoded' on the CL1GameStates component ->")
    _log("       switch on Result.State -> AddMovementInput per direction")
    _log("  3. (input) call 'Encode State' to stimulate for a chosen state")


if __name__ == "__main__":
    main()
