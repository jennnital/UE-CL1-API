// Copyright. Blueprint convenience accessor for the CL1 bridge subsystem.

#include "Cl1BridgeLibrary.h"
#include "Cl1BridgeSubsystem.h"
#include "Engine/Engine.h"
#include "Engine/GameInstance.h"
#include "Engine/World.h"

UCl1BridgeSubsystem* UCl1BridgeLibrary::GetCl1Bridge(const UObject* WorldContextObject)
{
	const UWorld* World = GEngine ? GEngine->GetWorldFromContextObject(WorldContextObject, EGetWorldErrorMode::LogAndReturnNull) : nullptr;
	const UGameInstance* GameInstance = World ? World->GetGameInstance() : nullptr;
	return GameInstance ? GameInstance->GetSubsystem<UCl1BridgeSubsystem>() : nullptr;
}
