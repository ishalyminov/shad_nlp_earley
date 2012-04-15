#!/usr/bin/python
################################################################################
#   * 10.04 - Fixed QTree printing. Thanks to Igor Shalyminov.
#   * 10.04 - Implemented proper backtracking and forest restoration. Thanks to Pavel Sergeev.
#   * 20.03 - Initial version.
################################################################################
# GLOSSARY
################################################################################
#   * Term
# Either terminal or non-terminal symbol.
#   * Production
# A right-hand side of a production rule; formally, a sequence of terms.
#   * Rule
# A set of all possible production rules, grouped by left-hand side.
#
# For example, in grammar:
#   S  -> NP VP
#   NP -> D N
#   NP -> John
#   D  -> the
#   D  -> a
#   N  -> cat
#   N  -> dog
#   ...
#
# "S", "NP", "VP", "D", "N", "John", "the", "a", "cat", "god"
#   are terms.
# [ "NP, "VP" ], [ "D", "N" ], [ "John" ], [ "the" ], [ "a" ], ...
#   are productions for productions rules (1) and (2) respectivelly.
# ("S", [ [ "NP" "VP" ] ]), ("NP", [ [ "D", "N" ], [ "John"] ]), ...
#   are rules.
   
class Production(object):
    def __init__(self, *terms):
        self.terms = terms

    def __len__(self):
        return len(self.terms)

    def __getitem__(self, index):
        return self.terms[index]

    def __iter__(self):
        return iter(self.terms)

    def __repr__(self):
        return " ".join(str(t) for t in self.terms)

    def __eq__(self, other):
        if not isinstance(other, Production):
            return False
        return self.terms == other.terms

    def __ne__(self, other):
        return not (self == other)

    def __hash__(self):
        return hash(self.terms)

class Rule(object):
    def __init__(self, name, *productions):
        self.name = name
        self.productions = list(productions)

    def __str__(self):
        return self.name

    def __repr__(self):
        return "%s -> %s" % (
            self.name,
            " | ".join(repr(p) for p in self.productions))

    def add(self, *productions):
        self.productions.extend(productions)

# State is a 3-tuple of a dotted rule, start column and end column.
class State(object):
    # A dotted rule is represented as a (name, production, dot_index) 3-tuple.
    def __init__(self, name, production, dot_index, start_column, end_column = None):
        self.name = name
        self.production = production
        self.dot_index = dot_index

        self.start_column = start_column
        self.end_column = end_column
        
        self.rules = [ term for term in self.production if isinstance(term, Rule) ]

    def __repr__(self):
        terms = [ str(term) for term in self.production ]
        terms.insert(self.dot_index, "$")

        return "%-5s -> %-23s [%2s-%-2s]" % (
            self.name,
            " ".join(terms),
            self.start_column,
            self.end_column)

    def __eq__(self, other):
        return \
            (self.name,  self.production,  self.dot_index,  self.start_column) == \
            (other.name, other.production, other.dot_index, other.start_column)

    def __ne__(self, other):
        return not (self == other)

    # Note that objects are hashed by (name, production), but not the whole state.
    def __hash__(self):
        return hash((self.name, self.production))

    def is_completed(self):
        return self.dot_index >= len(self.production)

    def get_next_term(self):
        if self.is_completed():
            return None
        return self.production[self.dot_index]

# Column is a list of states in a chart table.
class Column(object):
    def __init__(self, index, token):
        self.index = index
        self.token = token

        self.states = []
        self._predecessors = {}

    def __str__(self):
        return str(self.index)

    def __len__(self):
        return len(self.states)

    def __iter__(self):
        return iter(self.states)

    def __getitem__(self, index):
        return self.states[index]

    def add(self, state, predecessor = None):
        if state not in self._predecessors:
            self._predecessors[state] = set()
            state.end_column = self
            self.states.append(state)
        if predecessor is not None:
            self._predecessors[state].add(predecessor)
            

    def dump(self, only_completed = False):
        print " [%s] %r" % (self.index, self.token)
        print "=" * 40
        for s in self.states:
            if only_completed and not s.is_completed():
                continue
            print repr(s)
        print "=" * 40
        print
    
    def predecessors(self, state):
        return self._predecessors[state]

class Node(object):
    def __init__(self, value, children):
        self.value = value
        self.children = children

    def dump(self, level = 0):
        print "  " * level + str(self.value)
        for child in self.children:
            child.dump(level + 1)

# INTERNAL SUBROUTINES FOR EARLEY ALGORITHM
################################################################################

def predict(column, rule):
    for production in rule.productions:
        column.add(
            State(
                rule.name,
                production,
                0,
                column))

def scan(column, state, token):
    if token != column.token:
        return
    column.add(
        State(
            state.name,
            state.production,
            state.dot_index + 1,
            state.start_column), (state, None))

def complete(column, state):
    if not state.is_completed():
        return
    for prev_state in state.start_column:
        term = prev_state.get_next_term()
        if not isinstance(term, Rule):
            continue
        if term.name == state.name:
            column.add(
                State(
                    prev_state.name,
                    prev_state.production,
                    prev_state.dot_index + 1,
                    prev_state.start_column), (prev_state, state))

GAMMA_RULE = "GAMMA"

# ENTRY POINT FOR EARLEY ALGORITHM
################################################################################
def parse(starting_rule, text):
    text_with_indexes = enumerate([ None ] + text.lower().split())

    table = [ Column(i, token) for i, token in text_with_indexes ]
    table[0].add(State(GAMMA_RULE, Production(starting_rule), 0, table[0]))

    for i, column in enumerate(table):
        for state in column:
            if state.is_completed():
                complete(column, state)
            else:
                term = state.get_next_term()
                if isinstance(term, Rule):
                    predict(column, term)
                elif i + 1 < len(table):
                    scan(table[i + 1], state, term)
        
        # XXX(sandello): You can uncomment this line to see full dump of
        # the chart table.
        #
        # column.dump(only_completed = False)

    # Find Gamma rule in the last table column or fail otherwise.
    for state in table[-1]:
        if state.name == GAMMA_RULE and state.is_completed():
            return [tree for tree in build_trees(state, table)]
    else:
        return []

# AUXILIARY ROUTINES
################################################################################
def build_trees(state, table):
    for children in build_children(state, table, []):
        yield Node(state, [c for c in reversed(children)])

def build_children(state, table, prev_children):
    has_predecessor = False
    for predecessor, child in table[state.end_column.index].predecessors(state):
        has_predecessor = True
        if child is not None:
            for tree in build_trees(child, table):
                prev_children.append(tree)
                for children in build_children(predecessor, table, prev_children):
                    yield children
                prev_children.pop()
        else:
            for children in build_children(predecessor, table, prev_children):
                yield children
    if not has_predecessor:
        yield prev_children

def qtree(node):
    # http://yohasebe.com/rsyntaxtree/
    if node.value.name == GAMMA_RULE:
        return qtree(node.children[0])

    # These are subtrees in parse tree.
    lhs = list(child.value.name for child in node.children)
    # These are non-terminals from the grammar.
    rhs = list(term.name for term in node.value.production if isinstance(term, Rule))

    assert lhs == rhs

    idx = 0
    parts = []

    for term in node.value.production:
        if isinstance(term, Rule):
            parts.append(qtree(node.children[idx]))
            idx += 1
        else:
            parts.append(term)

    return "[{0} {1}]".format(node.value.name, " ".join(parts))

################################################################################

def load_grammar(iterable):
    non_terminals = dict()
    starting_rule = None

    def get_term(part):
        if part == part.lower():
            return part
        if part == part.upper():
            if part not in non_terminals:
                non_terminals[part] = Rule(part)
            return non_terminals[part]
        raise RuntimeError, "(unreachable)"

    for n, line in enumerate(iterable):
        parts = line.strip().split()

        for part in parts:
            if part != part.upper() and part != part.lower():
                raise RuntimeError, "Malformed line #{0}: Mixed-case for term '{1}'".format(n + 1, part)

        if len(parts) == 0:
            continue

        if len(parts) == 1:
            if parts[0] not in non_terminals:
                raise RuntimeError, "Malformed line #{0}: Unknown non-terminal '{1}'".format(n + 1, parts[0])
            else:
                starting_rule = parts[0]
                continue

        if parts[1] != "->":
            raise RuntimeError, "Malformed line #{0}: Second part have to be '->'".format(n + 1)

        lhs = get_term(parts[0])
        rhs = map(get_term, parts[2:])

        if not isinstance(lhs, Rule):
            raise RuntimeError, "Malformed line #{0}: Left-hand side have to be a non-terminal".format(n + 1)

        lhs.add(Production(*rhs))

    if starting_rule:
        return non_terminals[starting_rule]
    else:
        return non_terminals["S"]

################################################################################

if __name__ == "__main__":
    # You can specify grammar either by hard-coding it or by loading from file.
    # 
    # (Specifying grammar in code)
    #     SYM  = Rule("SYM", Production("a"))
    #     OP   = Rule("OP",  Production("+"), Production("*"))
    #     EXPR = Rule("EXPR")
    #     EXPR.add(Production(SYM))
    #     EXPR.add(Production(EXPR, OP, EXPR))
    #
    # (Loading grammar from file)
    #     g = load_grammar(open("a.txt"))

    g = load_grammar("""
        N -> hat
        N -> elephant
        N -> garden
        N -> apple
        N -> time
        N -> flight
        N -> banana
        N -> flies
        N -> boy
        N -> man
        N -> telescope

        NN -> john
        NN -> mary
        NN -> houston

        ADJ -> giant
        ADJ -> red

        D -> the
        D -> a
        D -> an

        V -> book
        V -> books
        V -> eat
        V -> eats
        V -> sleep
        V -> sleeps
        V -> give
        V -> gives
        V -> walk
        V -> walks
        V -> saw

        P -> with
        P -> in
        P -> on
        P -> at
        P -> through

        PR -> he
        PR -> she
        PR -> his
        PR -> her

        NP -> NN
        NP -> D N
        NP -> D ADJ N
        NP -> PR
        NP -> PR N
        NP -> NP PP

        PP -> P NP

        VP -> V
        VP -> V NP
        VP -> V NP NP
        VP -> VP PP

        S -> NP VP
        S -> VP
    """.splitlines())
    
    def parse_and_print(g, s):
        for tree in parse(g, s):
            print "-" * 80
            print qtree(tree)
            print
            tree.dump()
            print

    # phrases of your choice for testing parsing improvements
    parse_and_print(g, 'a boy eats a big apple')
