from MAP.pad_lexer import PadRangeParser, srange
import json, os


def int_to_row(row):
    return chr(ord('A')+row)

def row_to_int(row):
    return ord(row)-ord('A')


class Pad():
    """ Translates row, column notation to imaging position.
    Distances in um. 1um = 1e-6m
    """

    def __init__(self, *args):

        if len(args) == 1:
            row = args[0][0]
            col = args[0][1:]  # col can have multiple digits
        elif len(args) == 2:
            row, col = args
        else:
            raise ValueError(f'Pad __init__ error: Expected one or two arguments, got {len(args)} ({args})')

        if type(col) == str:
            col = int(col)

        if col not in self.cols:
            raise ValueError(f'Pad __init__ error: Column {col} invalid, must be in range {self.props["first_col"]}-{self.props["last_col"]}')

        if row not in self.rows:
            raise ValueError(f'Pad __init__ error: Row {row} invalid, must be in range {self.props["first_row"]}-{self.props["last_row"]}')

        # internal representation: zero indexed,
        self.row = row
        self.col = col

    def __eq__(self, __o: object) -> bool:
        return self.name == __o.name

    def __hash__(self) -> int:
        return self.name.__hash__()

    @property
    def name(self) -> str:
        return f'{self.row}{self.col}'

    # row_int, col_int are zero-indexed integers denoting the row and column of the pad.

    @property
    def y(self) -> float:
        return row_to_int(self.row)

    @property
    def x(self) -> float:
        return self.col-1

    @property
    def row_int(self) -> int:
        return row_to_int(self.row)

    @property
    def rows(self) -> list:
        return srange(self.props["first_row"], self.props["last_row"])

    @property
    def cols(self) -> list:
        return range(self.props["first_col"], self.props["last_col"]+1) # inclusive range

    # center coordinate of pad relative to feducial
    @property
    def pad_center(self) -> tuple: #[float, float]
        x = self.x * self.props["grid_pitch_x"] + self.props['feducial_offset_x']
        y = self.y * self.props["grid_pitch_y"] + self.props['feducial_offset_y']

        return (x, y)

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f'Pad({self.name})'

    @classmethod
    def range(cls, expression = ''):
        return [cls(p) for p in cls.parser.parse(expression)]

    @classmethod
    def range_in_traversal_order(cls, expression = ''):
        pads = cls.range(expression=expression)
        rows = sorted(set([p.row for p in pads]))
        row_groups = [[p for p in pads if p.row == r] for r in rows]
        reversed_row_groups = [g[::-1 if i%2 else 1] for i,g in enumerate(row_groups)]
        return [p for g in reversed_row_groups for p in g]

    @classmethod
    def from_point(cls, point, nikon=False):
        x,y = point

        if nikon:
            x = -x

        col = (x - cls.props['feducial_offset_x']) / cls.props['grid_pitch_x'] + 0.5 + 1 # one indexed
        row = (y - cls.props['feducial_offset_y']) / cls.props['grid_pitch_y'] + 0.5 # zero indexed

        col = min(max(int(col), cls.props['first_col']), cls.props['last_col'])
        row = min(max(int(row), row_to_int(cls.props['first_row'])), row_to_int(cls.props['last_row']))
        
        row_label = int_to_row(row)
        col_label = col
        
        return cls(row_label, col_label)  


def platform_props(filename):
    SCRIPT_DIRECTORY = os.path.dirname(__file__)  # absolute dir the script is in
    filepath = os.path.join(SCRIPT_DIRECTORY, 'sample_platforms', f'{filename}.json')
    with open(filepath) as file:
        data = file.read()
    props = json.loads(data)
    return props


class Pad_60_Well_Plate(Pad):

    props = platform_props('pad_60_well_plate')
    parser = PadRangeParser(props=props)

    def __init__(self, *args):
        super().__init__(*args)


class Pad_96_Well_Plate(Pad):
    
    props = platform_props('pad_96_well_plate')
    parser = PadRangeParser(props=props)

    def __init__(self, *args):
        super().__init__(*args)


class Pad_384_Well_Plate(Pad):
    
    props = platform_props('pad_384_well_plate')
    parser = PadRangeParser(props=props)

    def __init__(self, *args):
        super().__init__(*args)



def test(Pad):
    print(Pad('A', 1))
    print(Pad('F', 10))
    print(Pad('A', 1).pad_center)
    print(Pad('B', 2).pad_center)
    print(Pad('A2'))
    print(Pad('F10'))
    print(Pad.range('(A,C,F)2'))
    print(Pad.range('(A,B)(12,24)'))
    print(Pad.range('(A,F)(2,5)'))
    print(Pad.range('(1-3),(7-9)'))
    print(Pad.range())
    print(Pad.range_in_traversal_order())
    
    # if Pad is Pad_60_Well_Plate:
    #     assert(Pad.from_point([0,0]) == Pad('A1'))
    #     assert(Pad.from_point([-5500, 5500], nikon=True) == Pad('A1')) # A1
    #     assert(Pad.from_point([-6500, 6500], nikon=True) == Pad('A1')) # A1
    #     assert(Pad.from_point([-14380, 5900], nikon=True) == Pad('A2')) # A2
    #     assert(Pad.from_point([-87000, 15000], nikon=True) == Pad('B10')) # B10
    #     assert(Pad.from_point([5500, 5500], nikon=False) == Pad('A1')) # A1
    #     assert(Pad.from_point([6500, 6500], nikon=False) == Pad('A1')) # A1
    #     assert(Pad.from_point([15000, 6000], nikon=False) == Pad('A2')) # A2
    #     assert(Pad.from_point([87000, 15000], nikon=False) == Pad('B10')) # B10

    # assert(Pad("A1") == Pad("A1"))
    # assert(Pad("A1") != Pad("A2"))
    # assert(Pad("A1") != Pad("B1"))

    print(f'Hash: {Pad("A1").__hash__()}')


if __name__ == '__main__':
    # print('\nPad_60_Well_Plate')
    # test(Pad_60_Well_Plate)
    # print('\nPad_96_Well_Plate')
    # test(Pad_96_Well_Plate)
    print('\nPad_384_Well_Plate')
    test(Pad_384_Well_Plate)
