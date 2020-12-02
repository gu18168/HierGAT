from .utils import load_csv


def parse(pattern):
    attributes = {}

    for attr_str in pattern.split(';'):
        attr, fields = attr_str.split(':')
        attributes[attr] = fields.split(',')

    def decorate(cls):
        setattr(cls, "attributes", attributes)
        return cls

    return decorate

def involve(pattern):
    @parse(pattern)
    class Entity:
        def __init__(self, raw, id_prefix):
            self.id_prefix = id_prefix

            for attr, fields in self.attributes.items():
                value = ' '.join([str(raw[field]) for field in fields])
                if attr == 'id':
                    value = id_prefix + value
                setattr(self, attr, value)

        def to_string(self):
            return ' '.join([getattr(self, attr)
                             for attr in self.attributes.keys()])

        def to_attr(self, attr):
            if attr in self.attributes:
                return getattr(self, attr)
            raise ValueError("attr isn't found in Entity")

    return Entity

def get_entities(file_paths, pattern):
    Entity = involve(pattern)
    entities = [load_csv(file_path, lambda row: Entity(row, file_path.name))
                for file_path in file_paths]

    return entities, Entity