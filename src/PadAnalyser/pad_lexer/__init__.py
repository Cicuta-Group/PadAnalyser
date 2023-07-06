# -----------------------------------------------------------------------------
# example.py
#
# Example of using PLY To parse the following simple grammar.
#
#   expression : range SEP range
#              | range
#
#   range      : rows cols
#
#   rows       : ROW
#              | LPAREN ROW SEP ROW RPAREN
#              | LPAREN ROW RANGE ROW RPAREN
# 
#   cols       : COL
#              | LPAREN COL SEP COL RPAREN
#              | LPAREN COL RANGE COL RPAREN
#
# -----------------------------------------------------------------------------

import itertools
from typing import List
NoneType = type(None)

def srange(a, b):
    if ord(a) > ord(b) : a,b = b,a
    return [chr(i) for i in range(ord(a), ord(b)+1)]

def flatten(xs):
    for x in xs:
        if isinstance(x, List):
            yield from flatten(x)
        else:
            yield x


from ply import lex, yacc

class PadRangeParser:

    # --- Tokenizer

    # All tokens must be named in advance.
    tokens = ( 'RANGE', 'SEP', 'ROW', 'COL', 'LPAREN', 'RPAREN')

    # Ignored characters
    t_ignore = ' \t'

    # Token matching rules are written as regexs
    t_RANGE = r'-'
    t_SEP = r','
    t_ROW = r'[A-P]'
    t_COL = r'[1-9][0-9]?' # 1-24

    t_LPAREN = r'\('
    t_RPAREN = r'\)'

    # Ignored token with an action associated with it
    def t_ignore_newline(self, t):
        r'\n+'
        t.lexer.lineno += t.value.count('\n')

    # Error handler for illegal characters
    def t_error(self, t):
        print(f'Illegal character {t.value[0]!r}')
        t.lexer.skip(1)
        
    # --- Parser

    # Write functions for each grammar rule which is
    # specified in the docstring.
    def p_expression(self, p):
        '''
        expression : range SEP expression
        '''
        # p is a sequence that represents rule contents.
        #
        # expression : range SEP range
        #   p[0]     : p[1] p[2] p[3]
        # 
        exp = [p[1]], [p[3]]
        p[0] = list(flatten(exp))

    def p_expression_range(self, p):
        '''
        expression : range
        '''
        exp = [p[1]]
        p[0] = list(flatten(exp))

    def pads_from_rows_cols(self, rows, cols):
        rs = list(flatten([rows]))
        cs = list(flatten([cols]))
        return [f'{a}{b}' for a, b in list(itertools.product(rs, cs))]

    def p_range(self, p):
        '''
        range : rows cols
        '''
        p[0] = self.pads_from_rows_cols(p[1], p[2])

    def p_range_no_cols(self, p):
        '''
        range : rows
        '''
        cols = [f'{i}' for i in range(self.props['first_col'], self.props['last_col']+1)]
        p[0] = self.pads_from_rows_cols(p[1], cols)
        
    def p_range_no_rows(self, p):
        '''
        range : cols
        '''
        rows = srange(self.props['first_row'], self.props['last_row'])
        p[0] = self.pads_from_rows_cols(rows, p[1])

    def p_range_rows(self, p):
        '''
        rows : ROW
        '''
        p[0] = p[1]

    def p_range_cols(self, p):
        '''
        cols : COL
        '''
        p[0] = p[1]

    def p_range_rows_grouped(self, p):
        '''
        rows : LPAREN row_exp RPAREN
        '''
        p[0] = p[2]

    def p_range_cols_grouped(self, p):
        '''
        cols : LPAREN col_exp RPAREN
        '''
        p[0] = p[2]

    def p_range_row_exp(self, p):
        '''
        row_exp : ROW SEP row_exp
                | ROW RANGE ROW
        '''
        if p[2] == self.t_SEP:
            p[0] = [p[1], p[3]]
        elif p[2] == self.t_RANGE:
            p[0] = srange(p[1], p[3])

    def p_range_col_exp(self, p):
        '''
        col_exp : COL SEP col_exp
                | COL RANGE COL
        '''
        if p[2] == self.t_SEP:
            p[0] = [p[1], p[3]]
        elif p[2] == self.t_RANGE:
            p[0] = [f'{i}' for i in range(int(p[1]), int(p[3])+1)]

    def p_range_row(self, p):
        '''
        row_exp : ROW
        '''
        p[0] = p[1]

    def p_range_col(self, p):
        '''
        col_exp : COL
        '''
        p[0] = p[1]

    def p_error(self, p):
        print(f'Syntax error at {p.value!r}')


    def parse(self, expression):
        if not expression: 
            expression = f'({self.props["first_row"]}-{self.props["last_row"]})({self.props["first_col"]}-{self.props["last_col"]})'
        return self.yacc.parse(expression)

    def __init__(self, props):
        self.props = props # dict with keys
        self.lex = lex.lex(module=self)
        self.yacc = yacc.yacc(module=self)



def main():

    padRange = PadRangeParser(
        props = {
            'first_row': 'A',
            'last_row': 'F',
            'first_col': 1,
            'last_col': 10,
        }
    )

    tests = [
        'A1',
        '(A-C)3', # = A1...A10, B1...B10, C1...C10,
        '(A-C)(4,5)', # = A1, B1, C1,
        'A(1-3)', # = A1, A2, A3,
        '(A-C)(1,3)', # = A1, A3, B1, B3, C1, C3
        '(A-C)(10,12)', # = A10, A12, B10, B12, C10, C12
        '(A-C)(3-4)', # = A3, A4, B3, B4, C3, C4,
        'A1,B2,C3', # = A1, B2, C3,
        '(A,B)(1,2,3)',
        '(A,B,C)(1,2,3),(A-C)5',
        'A',
        '(A-B)',
        '1',
        '(2-3)',
    ]

    for test in tests:
        # Parse an expression
        print(f'{test} -> {padRange.parse(test)}')

if __name__ == '__main__':
    main()