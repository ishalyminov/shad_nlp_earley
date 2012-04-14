import re
import copy

CONSTRAINT_PATTERN = '<(.+)>\s*=\s*(.+)'
RULE_PATTERN = '[^<>]+\s+->\s+[^<>]'
COMMENT_PATTERN = '^#'

def new_empty_avm():
    return {'pointer': None, 'content': None}

def line_type(in_line):
    if len(re.findall(COMMENT_PATTERN, in_line)):
        return 'COMMENT'
    elif len(re.findall(CONSTRAINT_PATTERN, in_line)):
        return 'CONSTRAINT'
    elif len(re.findall(RULE_PATTERN, in_line)):
        return 'RULE'
    return 'UNK'

class FeatureStructureBuilder(object):
    def __init__(self):
        self.avm = new_empty_avm()
    
    def initialize_structures(self, in_constraint_string):
        current_avm = self.avm
        previous_avm = None
        target_feature = None
    
        for term in in_constraint_string.strip().split():
            if not current_avm['content']:
                current_avm['content'] = {}
            if term not in current_avm['content']:
                current_avm['content'][term] = new_empty_avm()
            target_feature = term
            previous_avm = current_avm
            current_avm = current_avm['content'][term]
        # returns the deepest avm (previous_avm['content'][target_feature]) that is left empty
        return (previous_avm, target_feature)
        
    def process_constraint(self, in_constraint_string):
        constraint_matches = re.findall(CONSTRAINT_PATTERN, in_constraint_string)
        if not constraint_matches:
            return
        constraint_matches = constraint_matches[0]
        
        (lhs_avm, lhs_term) = self.initialize_structures(constraint_matches[0])
    
        # determining whether rhs is a complex feature structure
        # or an atomic value
        complex_rhs = re.findall('<(.+)>', constraint_matches[1])
        # complex feature structure case
        if len(complex_rhs):
            (rhs_avm, rhs_term) = self.initialize_structures(complex_rhs[0])
            # linking
            lhs_avm['content'][lhs_term] = rhs_avm['content'][rhs_term]
        # atomic value case
        else:
            lhs_avm['content'][lhs_term] = new_empty_avm()
            lhs_avm['content'][lhs_term]['content'] = constraint_matches[1].strip()
    def get_avm(self):
        return self.avm

def follow_path(in_avm, in_name):
    avm = dereference(in_avm)
    if isinstance(avm['content'], dict) and in_name in avm['content']:
        return avm['content'][in_name]
    return avm

def dereference(in_avm):
    while in_avm and in_avm['pointer']:
        in_avm = in_avm['pointer']
    return in_avm

def unify(in_lhs, in_rhs):
    lhs = dereference(in_lhs)
    rhs = dereference(in_rhs)
    
    if lhs['content'] == rhs['content'] or not lhs['content']:
        lhs['pointer'] = rhs
        return rhs
    elif not rhs['content']:
        rhs['pointer'] = lhs
        return lhs
    elif isinstance(lhs['content'], dict) and isinstance(rhs['content'], dict):
        rhs['pointer'] = lhs
        for feature in rhs['content']:
            if feature not in lhs['content']:
                lhs['content'][feature] = new_empty_avm()
            unification_result = unify(lhs['content'][feature], rhs['content'][feature])
            if unification_result == 'FAIL':
                return unification_result
            lhs['content'][feature] = unification_result
        return lhs
    return 'FAIL'

def unify_states(in_lhs_avm, in_rhs_avm, in_name):
    lhs_avm_copy = copy.deepcopy(in_lhs_avm)
    rhs_avm_copy = copy.deepcopy(in_rhs_avm)
    
    if isinstance(rhs_avm_copy['content'], dict) and in_name in rhs_avm_copy['content']:
        result = unify(follow_path(lhs_avm_copy, in_name), follow_path(rhs_avm_copy, in_name))
        if result == 'FAIL':
            return result
    return rhs_avm_copy

if __name__ =='__main__':
    grammar = """
        S -> NP VP
        <NP HEAD AGR> = <VP HEAD AGR>
        <S HEAD> = <VP HEAD>
        
        <N HEAD AGR NUMBER> = sg
        <N HEAD AGR PERSON> = 3rd

        V -> sleep
        <V HEAD AGR NUMBER> = pl
        
        V -> talks
        <V HEAD AGR NUMBER> = sg
        
        NP -> N
        <NP HEAD AGR> = <N HEAD AGR>

        VP -> V
        <VP HEAD AGR> = <V HEAD AGR>
    """.splitlines()
    builder = FeatureStructureBuilder()
    for line in grammar:
        builder.process_constraint(line)
        