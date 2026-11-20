"""
Boolean Query Parser and Filtering Engine.
Supports AND, OR, NOT operations, parentheses grouping, and multi-keyword queries.
Evaluates query against an inverted index to produce matching document subsets.
"""
import re
from typing import Set, Dict, List
from .preprocessor import preprocess_text

class BooleanQueryParser:
    def __init__(self, documents: List[str]):
        """
        Builds inverted index: {preprocessed_term: set(doc_indices)}
        """
        self.doc_count = len(documents)
        self.inverted_index: Dict[str, Set[int]] = {}
        self._build_inverted_index(documents)

    def _build_inverted_index(self, documents: List[str]):
        for idx, doc in enumerate(documents):
            terms = set(preprocess_text(doc))
            for term in terms:
                if term not in self.inverted_index:
                    self.inverted_index[term] = set()
                self.inverted_index[term].add(idx)

    def is_boolean_query(self, query: str) -> bool:
        """
        Checks if query contains Boolean operators (AND, OR, NOT).
        """
        tokens = query.split()
        return any(t in ("AND", "OR", "NOT") for t in tokens)

    def _tokenize_boolean_query(self, query: str) -> List[str]:
        """
        Splits expression preserving parentheses and Boolean operators.
        """
        pattern = r'\(|\)|\bAND\b|\bOR\b|\bNOT\b|[^\s()]+'
        matches = re.findall(pattern, query)
        return matches

    def _shunting_yard(self, tokens: List[str]) -> List[str]:
        """
        Converts infix Boolean tokens to postfix (Reverse Polish Notation).
        Precedence: NOT > AND > OR.
        Inserts implicit AND between adjacent terms if missing operator.
        """
        precedence = {'NOT': 3, 'AND': 2, 'OR': 1}
        output = []
        op_stack = []
        
        # Insert implicit AND where appropriate
        expanded_tokens = []
        for i, token in enumerate(tokens):
            expanded_tokens.append(token)
            if i + 1 < len(tokens):
                next_tok = tokens[i + 1]
                # If current is term or ')' and next is term or '(' or 'NOT', add implicit AND
                is_curr_term = token not in ('AND', 'OR', 'NOT', '(')
                is_next_term = next_tok not in ('AND', 'OR', ')')
                if (is_curr_term or token == ')') and (is_next_term or next_tok in ('(', 'NOT')):
                    expanded_tokens.append('AND')

        for token in expanded_tokens:
            if token in ('AND', 'OR', 'NOT'):
                while (op_stack and op_stack[-1] != '(' and 
                       precedence.get(op_stack[-1], 0) >= precedence[token]):
                    output.append(op_stack.pop())
                op_stack.append(token)
            elif token == '(':
                op_stack.append(token)
            elif token == ')':
                while op_stack and op_stack[-1] != '(':
                    output.append(op_stack.pop())
                if op_stack and op_stack[-1] == '(':
                    op_stack.pop()
            else:
                output.append(token)
                
        while op_stack:
            output.append(op_stack.pop())
            
        return output

    def evaluate(self, query: str) -> Set[int]:
        """
        Evaluates a Boolean query and returns set of matching document indices.
        If no Boolean operators are present, defaults to union/intersection of multi-keywords.
        """
        all_docs = set(range(self.doc_count))
        tokens = self._tokenize_boolean_query(query)
        
        if not tokens:
            return set()

        # If it's a plain query without explicit operators, return union of all matching term docs
        has_explicit_bool = any(t in ('AND', 'OR', 'NOT') for t in tokens)
        if not has_explicit_bool:
            query_terms = preprocess_text(query)
            if not query_terms:
                return all_docs
            result = set()
            for t in query_terms:
                result.update(self.inverted_index.get(t, set()))
            return result

        postfix = self._shunting_yard(tokens)
        eval_stack: List[Set[int]] = []

        for token in postfix:
            if token == 'NOT':
                s = eval_stack.pop() if eval_stack else set()
                eval_stack.append(all_docs - s)
            elif token == 'AND':
                right = eval_stack.pop() if eval_stack else set()
                left = eval_stack.pop() if eval_stack else set()
                eval_stack.append(left & right)
            elif token == 'OR':
                right = eval_stack.pop() if eval_stack else set()
                left = eval_stack.pop() if eval_stack else set()
                eval_stack.append(left | right)
            else:
                # Preprocess term
                stemmed_terms = preprocess_text(token)
                if stemmed_terms:
                    term_doc_set = set()
                    for st in stemmed_terms:
                        term_doc_set.update(self.inverted_index.get(st, set()))
                    eval_stack.append(term_doc_set)
                else:
                    eval_stack.append(set())

        return eval_stack[0] if eval_stack else set()
