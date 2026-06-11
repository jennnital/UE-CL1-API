// Copyright. Build rules for the UeCl1Api runtime module.

using UnrealBuildTool;

public class UeCl1Api : ModuleRules
{
	public UeCl1Api(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = ModuleRules.PCHUsageMode.UseExplicitOrSharedPCHs;

		PublicDependencyModuleNames.AddRange(new string[]
		{
			"Core",
			"CoreUObject",
			"Engine",
			"Sockets",
			"Networking"
		});

		PrivateDependencyModuleNames.AddRange(new string[] { });
	}
}
