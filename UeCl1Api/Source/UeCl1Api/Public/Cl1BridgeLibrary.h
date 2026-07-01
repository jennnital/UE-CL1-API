// Copyright. Blueprint convenience accessor for the CL1 bridge subsystem.

#pragma once

#include "CoreMinimal.h"
#include "Kismet/BlueprintFunctionLibrary.h"
#include "Cl1BridgeLibrary.generated.h"

class UCl1BridgeSubsystem;

UCLASS()
class UECL1API_API UCl1BridgeLibrary : public UBlueprintFunctionLibrary
{
	GENERATED_BODY()

public:
	/** Fetch the CL1 bridge for the current game instance. Valid for the lifetime of the game instance. */
	UFUNCTION(BlueprintPure, Category = "CL1", meta = (WorldContext = "WorldContextObject"))
	static UCl1BridgeSubsystem* GetCl1Bridge(const UObject* WorldContextObject);
};
