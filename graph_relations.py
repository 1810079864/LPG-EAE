"""Stable relation vocabulary shared by offline graph caching and the model."""

DEPENDENCY_RELATIONS = (
    'nsubj', 'nsubjpass', 'dobj', 'iobj', 'csubj', 'ccomp', 'xcomp',
    'root', 'aux', 'auxpass', 'nmod', 'amod', 'advmod', 'nummod',
    'appos', 'advcl', 'acl', 'relcl', 'prep', 'pobj', 'obj', 'obl',
    'det', 'compound', 'punct', 'conj', 'cc', 'mark', 'case', 'cop',
    'neg', 'poss', 'attr', 'acomp', 'oprd', 'parataxis', 'dep',
    'expl', 'discourse', 'fixed', 'flat', 'list', 'orphan',
    'vocative', 'dislocated', 'goeswith', 'reparandum',
)

EDGE_TYPE_NAMES = ('padding',) + tuple(
    'dep:{}:{}'.format(relation, direction)
    for relation in ('other',) + DEPENDENCY_RELATIONS
    for direction in ('forward', 'reverse')
) + ('coreference', 'trigger_group', 'self_loop')
EDGE_TYPE_TO_ID = {
    name: index for index, name in enumerate(EDGE_TYPE_NAMES)
}


def dependency_type(relation, reverse=False):
    """Map unknown dependency labels to an explicit fallback type."""
    relation = str(relation).lower()
    if relation not in DEPENDENCY_RELATIONS:
        relation = 'other'
    direction = 'reverse' if reverse else 'forward'
    return 'dep:{}:{}'.format(relation, direction)
