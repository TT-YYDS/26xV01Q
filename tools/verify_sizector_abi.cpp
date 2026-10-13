// Compile against the supplied 2.776 headers, without linking or opening hardware.
#include <string>
#include <map>
#include <cstddef>
#include "MPSizectorS_Types.h"

using namespace mpapi;
static_assert(sizeof(void*) == 8, "x64 required");
static_assert(sizeof(MPSizectorS_DeviceInfoStructType) == 192, "DeviceInfo ABI changed");
static_assert(sizeof(MPSizectorS_HeadPackDataInfoStructType) == 32, "DataInfo ABI changed");
static_assert(sizeof(MPSizectorS_HeadPackStructType) == 2048, "HeadPack ABI changed");
static_assert(alignof(MPSizectorS_HeadPackStructType) == 8, "HeadPack alignment changed");
static_assert(offsetof(MPSizectorS_HeadPackStructType, DataInfo) == 192, "DataInfo offset changed");
static_assert(offsetof(MPSizectorS_HeadPackStructType, BasicSettings) == 224, "Settings offset changed");
static_assert(sizeof(MPSizectorS_BasicSettingsStructType) == 16, "BasicSettings ABI changed");
static_assert(offsetof(MPSizectorS_HeadPackStructType, ExposureSettings) == 240, "Frame exposure offset changed");
static_assert(offsetof(MPSizectorS_BasicSettingsStructType, WorkingMode) == 0, "Mode offset changed");
static_assert(sizeof(MPSizectorS_DataFrameUndefinedStruct) == 2056, "Frame ABI changed");
static_assert(offsetof(MPSizectorS_DataFrameUndefinedStruct, Data) == 2048, "Frame pointer offset changed");
static_assert(sizeof(MPSizectorS_DataPointFloat3DStruct) == 16, "Point ABI changed");
static_assert(offsetof(MPSizectorS_DataPointFloat3DStruct, X) == 4, "Point coordinate offset changed");
static_assert(sizeof(MPSizectorS_ExposureBasicSettingStructType) == 8, "ExposureBasic ABI changed");
static_assert(sizeof(MPSizectorS_ExposureSettingsStructType) == 32, "ExposureSettings ABI changed");
static_assert(offsetof(MPSizectorS_ExposureSettingsStructType, UserGain) == 8, "Gain offset changed");
static_assert(offsetof(MPSizectorS_ExposureSettingsStructType, ExposureIntensity3D_1st) == 12, "Exposure offset changed");
static_assert(offsetof(MPSizectorS_ExposureSettingsStructType, MultiHeadExpoVarMode) == 28, "Multihead offset changed");
