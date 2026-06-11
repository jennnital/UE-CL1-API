using UnrealBuildTool;

public class AssembloidAgency : ModuleRules
{
	public AssembloidAgency(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

		PublicDependencyModuleNames.AddRange(new string[]
		{
			"Core",
			"CoreUObject",
			"Engine",
			"Sockets",
			"Networking",
			// From getnamo's UDP-Unreal plugin. Make sure that plugin is installed
			// and enabled in your .uproject / .uplugin.
			"UDPWrapper"
		});

		PrivateDependencyModuleNames.AddRange(new string[] { });
	}
}
