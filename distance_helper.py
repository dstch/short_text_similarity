class distance_helper(object):
    def __init__(self):
        pass

    @staticmethod
    def cos_distance(vector1, vector2):
        dot_product = 0.0
        normA = 0.0
        normB = 0.0
        for a, b in zip(vector1, vector2):
            dot_product += a * b
            normA += a ** 2
            normB += b ** 2
        if normA == 0.0 or normB == 0.0:
            return None
        else:
            return dot_product / ((normA * normB) ** 0.5)

    @staticmethod
    def euclidean_distance(vector1, vector2):
        d = 0
        for a, b in zip(vector1, vector2):
            d += (a - b) ** 2
        return d ** 0.5
