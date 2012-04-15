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

import constraint
import example

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

class ConstrainedProduction(Production):
    def __init__(self, feature_structure, *terms):
        Production.__init__(self, *terms)
        self.avm = feature_structure
    def __eq__(self, other):
        if not isinstance(other, ConstrainedProduction):
            return False
        return self.terms == other.terms and self.avm == other.avm

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

class ConstrainedState(State):
    def __init__(self, name, production, dot_index, constraint, start_column, end_column = None):
        State.__init__(self, name, production, dot_index, start_column, end_column)
        self.constraint = constraint

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

class GrammarBuilder(object):
    def __init__(self):
        self.feature_builder = constraint.FeatureStructureBuilder()
    
    def process_line(self, in_number, in_line):
        line_type = constraint.line_type(in_line.strip())
        if line_type == 'CONSTRAINT':
            self.process_constraint(in_number, in_line)
            self.last_production.avm = self.feature_builder.avm
        elif line_type == 'RULE':
            self.feature_builder = constraint.FeatureStructureBuilder()
            self.process_rule(in_number, in_line)
    
    def process_constraint(self, in_number, in_constraint):
        if not self.last_production:
            raise RuntimeError, 'Constraint line goes before rule. Grammar parse fail at line %d' % in_number
        self.feature_builder.process_constraint(in_constraint)
    
    def get_term(self, part):
        if part == part.lower():
            return part
        if part == part.upper():
            if part not in self.non_terminals:
                self.non_terminals[part] = Rule(part)
            return self.non_terminals[part]
        raise RuntimeError, "(unreachable)"
        
    def process_rule(self, in_number, in_line):
        line = in_line
        n = in_number
        parts = line.strip().split()
    
        for part in parts:
            if part != part.upper() and part != part.lower():
                raise RuntimeError, "Malformed line #{0}: Mixed-case for term '{1}'".format(n + 1, part)
    
        if len(parts) == 0:
            return
    
        if len(parts) == 1:
            if parts[0] not in self.non_terminals:
                raise RuntimeError, "Malformed line #{0}: Unknown non-terminal '{1}'".format(n + 1, parts[0])
            else:
                self.starting_rule = parts[0]
                return
    
        if parts[1] != "->":
            raise RuntimeError, "Malformed line #{0}: Second part have to be '->'".format(n + 1)
    
        lhs = self.get_term(parts[0])
        rhs = map(self.get_term, parts[2:])
    
        if not isinstance(lhs, Rule):
            raise RuntimeError, "Malformed line #{0}: Left-hand side have to be a non-terminal".format(n + 1)
        
        rhs_prod = ConstrainedProduction(constraint.new_empty_avm(), *rhs)
        lhs.add(rhs_prod)
        self.last_production = rhs_prod
        
    def load_grammar(self, iterable):
        self.non_terminals = dict()
        self.starting_rule = None
        self.last_production = None
        
        for n, line in enumerate(iterable):
            self.process_line(n, line)
        self.last_production.avm = self.feature_builder.get_avm()
            
        if self.starting_rule:
            return self.non_terminals[starting_rule]
        else:
            return self.non_terminals["S"]

# INTERNAL SUBROUTINES FOR EARLEY ALGORITHM
################################################################################

def predict(column, rule):
    for production in rule.productions:
        column.add(
            ConstrainedState(
                rule.name,
                production,
                0,
                production.avm,
                column))

def scan(column, state, token):
    if token != column.token:
        return
    column.add(
        ConstrainedState(
            state.name,
            state.production,
            state.dot_index + 1,
            state.constraint,
            state.start_column), (state, None))

def complete(column, state):
    if not state.is_completed():
        return
    for prev_state in state.start_column:
        term = prev_state.get_next_term()
        if not isinstance(term, Rule):
            continue
        if term.name == state.name:
            unifying = constraint.unify_states(state.constraint, prev_state.constraint, state.name)
            if unifying is not 'FAIL':
                column.add(
                    ConstrainedState(
                        prev_state.name,
                        prev_state.production,
                        prev_state.dot_index + 1,
                        unifying,
                        prev_state.start_column), (prev_state, state))

GAMMA_RULE = "GAMMA"

# ENTRY POINT FOR EARLEY ALGORITHM
################################################################################
def parse(starting_rule, text):
    import string
    splitted = string.replace(text.lower(), ',', ' , ').split()
    text_with_indexes = enumerate([ None ] + splitted)

    table = [ Column(i, token) for i, token in text_with_indexes ]
    
    gamma_prod = ConstrainedProduction(constraint.new_empty_avm(), starting_rule)
    table[0].add(ConstrainedState(GAMMA_RULE, gamma_prod, 0, gamma_prod.avm, table[0]))

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

    grammar_builder = GrammarBuilder()
    g = grammar_builder.load_grammar(open('grammar.txt'))
    
    def parse_and_print(g, s):
        trees = parse(g, s)
        for tree in trees:
            print "-" * 80
            print qtree(tree)
            print
            tree.dump()
            print
        return len(trees)

    for example_file in example.EXAMPLE_FILES:
        (header, lines) = example.load_examples(open(example_file))
        if not header or not lines:
            raise ValueError, 'invalid example file %s' % example_file
        print header
        print lines
        for line in lines:
            if not parse_and_print(g, line):
                print '*', line

