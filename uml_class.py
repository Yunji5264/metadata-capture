import json

class DS_Spatial_Scope:
    def __init__(self, spatialScopeLevel, spatialScope):
        self.spatialScopeLevel = spatialScopeLevel
        self.spatialScope = spatialScope

    def to_dict(self):
        return {
            "spatialScopeLevel": self.spatialScopeLevel,
            "spatialScope": self.spatialScope
        }

    @classmethod
    def from_dict(cls, data):
        return cls(data["spatialScopeLevel"], data["spatialScope"])


class DS_Temporal_Scope:
    def __init__(self, temporalScopeLevel, temporalScopeStart, temporalScopeEnd):
        self.temporalScopeLevel = temporalScopeLevel
        self.temporalScopeStart = temporalScopeStart
        self.temporalScopeEnd = temporalScopeEnd

    def to_dict(self):
        return {
            "temporalScopeLevel": self.temporalScopeLevel,
            "temporalScopeStart": self.temporalScopeStart,
            "temporalScopeEnd": self.temporalScopeEnd
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            data["temporalScopeLevel"],
            data["temporalScopeStart"],
            data["temporalScopeEnd"]
        )


class Theme:
    def __init__(self, themeCode, themeDescription):
        self.themeCode = themeCode
        self.themeDescription = themeDescription

    def to_dict(self):
        return {
            "themeCode": self.themeCode,
            "themeDescription": self.themeDescription
        }

    @classmethod
    def from_dict(cls, data):
        return cls(data["themeCode"], data["themeDescription"])


# Data_Content
class Data_Content:
    def __init__(self, dataName, dataDescription, dataType):
        self.dataName = dataName
        self.dataDescription = dataDescription
        self.dataType = dataType

    def to_dict(self):
        return {
            "dataName": self.dataName,
            "dataDescription": self.dataDescription,
            "dataType": self.dataType
        }

    @classmethod
    def from_dict(cls, data):
        return cls(data["dataName"], data["dataDescription"], data["dataType"])


# subclass：Parameter
class Parameter(Data_Content):
    def __init__(self, dataName, dataDescription, dataType, parameterCode):
        super().__init__(dataName, dataDescription, dataType)
        self.parameterCode = parameterCode

    def to_dict(self):
        data = super().to_dict()
        data["parameterCode"] = self.parameterCode
        return data

    @classmethod
    def from_dict(cls, data):
        return cls(
            data["dataName"],
            data["dataDescription"],
            data["dataType"],
            data["parameterCode"]
        )


# subclass：Spatial_Parameter
class Spatial_Parameter(Parameter):
    def __init__(self, dataName, dataDescription, dataType, parameterCode, spatialLevel):
        super().__init__(dataName, dataDescription, dataType, parameterCode)
        self.spatialLevel = spatialLevel

    def to_dict(self):
        data = super().to_dict()
        data["spatialLevel"] = self.spatialLevel
        return data

    @classmethod
    def from_dict(cls, data):
        return cls(
            data["dataName"],
            data["dataDescription"],
            data["dataType"],
            data["parameterCode"],
            data["spatialLevel"]
        )


# subclass：Temporal_Parameter
class Temporal_Parameter(Parameter):
    def __init__(self, dataName, dataDescription, dataType, parameterCode, temporalLevel):
        super().__init__(dataName, dataDescription, dataType, parameterCode)
        self.temporalLevel = temporalLevel

    def to_dict(self):
        data = super().to_dict()
        data["temporalLevel"] = self.temporalLevel
        return data

    @classmethod
    def from_dict(cls, data):
        return cls(
            data["dataName"],
            data["dataDescription"],
            data["dataType"],
            data["parameterCode"],
            data["temporalLevel"]
        )


# subclass：Complementary_Information
class Complementary_Information(Data_Content):
    def __init__(self, dataName, dataDescription, dataType, ciCode, related_parameter, theme):
        super().__init__(dataName, dataDescription, dataType)
        self.ciCode = ciCode
        self.related_parameter = related_parameter
        self.theme = theme

    def to_dict(self):
        data = super().to_dict()
        data["ciCode"] = self.ciCode
        if isinstance(self.related_parameter, (Spatial_Parameter, Temporal_Parameter)):
            data["related_parameter"] = self.related_parameter.to_dict()
        else:
            data["related_parameter"] = None  # Safely handle None or unrecognized types
        if isinstance(self.theme, Theme):
            data["theme"] = self.theme.to_dict()
        else:
            data["theme"] = None  # Safely handle None or unrecognized types
        return data

    @classmethod
    def from_dict(cls, data):
        related_parameter_data = data.get('related_parameter')
        related_parameter = None
        if related_parameter_data:
            if 'spatialLevel' in related_parameter_data:
                related_parameter = Spatial_Parameter.from_dict(related_parameter_data)
            elif 'temporalLevel' in related_parameter_data:
                related_parameter = Temporal_Parameter.from_dict(related_parameter_data)
        theme = Theme.from_dict(data["theme"]) if "theme" in data else None

        return cls(
            data["dataName"],
            data["dataDescription"],
            data["dataType"],
            data["ciCode"],
            related_parameter,
            theme
        )

# subclass：Existing_Indicator
class Existing_Indicator(Data_Content):
    def __init__(self, dataName, dataDescription, dataType, indicatorCode, theme):
        super().__init__(dataName, dataDescription, dataType)
        self.indicatorCode = indicatorCode
        self.theme = theme

    def to_dict(self):
        data = super().to_dict()
        data["indicatorCode"] = self.indicatorCode
        if isinstance(self.theme, Theme):
            data["related_parameter"] = self.theme.to_dict()
        else:
            data["related_parameter"] = None  # Safely handle None or unrecognized types
        return data

    @classmethod
    def from_dict(cls, data):
        return cls(
            data["dataName"],
            data["dataDescription"],
            data["dataType"],
            data["indicatorCode"],
            data["theme"]
        )

class Dataset:
    def __init__(
        self,
        title,
        description,
        dataFormat,
        fileType,
        updateFrequency,
        sourceName,
        sourceType,
        sourceAddress,
        spatialGranularity,
        spatialScope,       # DS_Spatial_Scope
        temporalGranularity,
        temporalScope,      # DS_Temporal_Scope
        theme,              # Theme
        data_content
    ):
        self.title = title
        self.description = description
        self.dataFormat = dataFormat
        self.fileType = fileType
        self.updateFrequency = updateFrequency
        self.sourceName = sourceName
        self.sourceType = sourceType
        self.sourceAddress = sourceAddress
        self.spatialGranularity = spatialGranularity
        self.spatialScope = spatialScope                    # DS_Spatial_Scope object
        self.temporalGranularity = temporalGranularity
        self.temporalScope = temporalScope                  # DS_Temporal_Scope object
        self.theme = theme                                  # Theme object
        self.data_content = data_content                    # List[Data_Content or subclasses]

    def to_dict(self):
        return {
            "title": self.title,
            "description": self.description,
            "dataFormat": self.dataFormat,
            "fileType": self.fileType,
            "updateFrequency": self.updateFrequency,
            "sourceName": self.sourceName,
            "sourceType": self.sourceType,
            "sourceAddress": self.sourceAddress,
            "spatialGranularity": self.spatialGranularity,
            "spatialScope": self.spatialScope.to_dict(),
            "temporalGranularity": self.temporalGranularity,
            "temporalScope": self.temporalScope.to_dict(),
            "theme": self.theme.to_dict(),
            "data_content": [content.to_dict() for content in self.data_content]
        }

    @classmethod
    def from_dict(cls, data):
        spatial_scope = DS_Spatial_Scope.from_dict(data["spatialScope"])
        temporal_scope = DS_Temporal_Scope.from_dict(data["temporalScope"])
        theme = Theme.from_dict(data["theme"])
        data_content = [cls._map_to_subclass(content) for content in data["data_content"]]

        return cls(
            title=data["title"],
            description=data["description"],
            dataFormat=data["dataFormat"],
            fileType=data["fileType"],
            updateFrequency=data["updateFrequency"],
            sourceName=data["sourceName"],
            sourceType=data["sourceType"],
            sourceAddress=data["sourceAddress"],
            spatialGranularity=data["spatialGranularity"],
            spatialScope=spatial_scope,
            temporalGranularity=data["temporalGranularity"],
            temporalScope=temporal_scope,
            theme=theme,
            data_content=data_content
        )

    @staticmethod
    def _map_to_subclass(data):
        if "parameterCode" in data and "spatialLevel" in data:
            return Spatial_Parameter.from_dict(data)
        elif "parameterCode" in data and "temporalLevel" in data:
            return Temporal_Parameter.from_dict(data)
        elif "ciCode" in data:
            return Complementary_Information.from_dict(data)
        elif "indicatorCode" in data:
            return Existing_Indicator.from_dict(data)
        elif "parameterCode" in data:
            return Parameter.from_dict(data)
        else:
            return Data_Content.from_dict(data)

    def save_to_json(self, file_path):
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=4)

    @classmethod
    def load_from_json(cls, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)