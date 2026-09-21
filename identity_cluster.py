"""Bounded, conservative person grouping using original vectors, not ANN scores."""
from collections import defaultdict

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


def unit_vectors(points, name):
    rows = []
    for p in points:
        v = np.asarray((p.get('vector') or {}).get(name, []), dtype=np.float32)
        if v.ndim != 1 or not v.size or not np.isfinite(v).all():
            raise ValueError(f"Missing/invalid {name} vector: {p['id']}")
        norm = np.linalg.norm(v)
        if norm <= 1e-12:
            raise ValueError(f"Zero {name} vector: {p['id']}")
        rows.append(v / norm)
    return np.stack(rows)


def score_matrices(points, primary, secondary):
    x = unit_vectors(points, primary)
    y = unit_vectors(points, secondary)
    return np.clip(x @ x.T, -1, 1), np.clip(y @ y.T, -1, 1)


def identity_constraints(points, a, b, threshold, secondary_threshold):
    # Conservative cannot-link: distinct detections in one source image.
    # Overlapping duplicate detections may stay split, intentionally.
    allowed = (a >= threshold) & (b >= secondary_threshold)
    by_image = defaultdict(list)
    for i, p in enumerate(points):
        payload = p.get('payload') or {}
        if payload.get('media_type') != 'image':
            raise ValueError('Identity-safe mode currently supports image points only.')
        key = payload.get('image_id')
        if not key:
            raise ValueError(f"Missing image_id: {p['id']}")
        by_image[key].append(i)
    for indices in by_image.values():
        allowed[np.ix_(indices, indices)] = False
    np.fill_diagonal(allowed, False)
    return allowed


def mutual_edges(a, allowed, k):
    neighbors = np.zeros_like(allowed)
    for i in range(len(a)):
        candidates = np.flatnonzero(allowed[i])
        order = np.argsort(-a[i, candidates], kind='stable')[:k]
        neighbors[i, candidates[order]] = True
    rows, cols = np.where(np.triu(neighbors & neighbors.T, 1))
    return list(zip(rows.tolist(), cols.tolist())), a[rows, cols].tolist()


def refine_membership(membership, a, b, allowed):
    """Complete-link split: every retained pair must satisfy both vector gates."""
    groups = defaultdict(list)
    for i, cid in enumerate(membership):
        groups[cid].append(i)
    result = [-1] * len(membership)
    next_id = 0
    for indices in groups.values():
        if len(indices) == 1:
            labels = [1]
        else:
            ix = np.ix_(indices, indices)
            # Valid pair distances <= 1, forbidden pairs = 2.
            distances = np.maximum((1 - a[ix]) / 2, (1 - b[ix]) / 2)
            distances[~allowed[ix]] = 2.0
            np.fill_diagonal(distances, 0)
            tree = linkage(squareform(distances, checks=False), method='complete')
            labels = fcluster(tree, t=1.0, criterion='distance')
        remap = {}
        for i, label in zip(indices, labels):
            if label not in remap:
                remap[label] = next_id
                next_id += 1
            result[i] = remap[label]
    return result


def fetch_points(q, collection, ids, primary, secondary):
    points = {}
    for start in range(0, len(ids), 128):
        reply = q.post(f'/collections/{collection}/points', {
            'ids': list(ids[start:start + 128]),
            'with_vector': list(dict.fromkeys([primary, secondary])),
            'with_payload': ['media_type', 'image_id'],
        })
        for p in reply['result']:
            points[str(p['id'])] = p
    return [points[str(pid)] for pid in ids]
