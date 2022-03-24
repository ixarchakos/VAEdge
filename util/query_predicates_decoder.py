

def _query_predicates():
    predicates = [
        "FIND MIN(CAR, 1) AND MIN(TRUCK, 1)",
        "",
    ]
    return predicates


def _dictionary():
    special_words = ["FIND", "MIN", "MAX", "RIGHT", "LEFT"]
    object_words = ["CAR", "TRUCK", "BUS", "HUMAN"]
    return special_words, object_words


def decoder():
    special_words, object_words = _dictionary()
    predicates = _query_predicates()

