#include "pbrt/euclidean_space/normal3f.h"
#include "pbrt/shapes/loop_subdivide.h"
#include <map>
#include <memory>
#include <set>

namespace {
struct SDFace;
struct SDVertex;

// LoopSubdiv Macros
#define NEXT(i) (((i) + 1) % 3)
#define PREV(i) (((i) + 2) % 3)

// LoopSubdiv Local Structures
struct SDVertex {
    // SDVertex Constructor
    SDVertex(const Point3f &p = Point3f(0, 0, 0)) : p(p) {}

    // SDVertex Methods
    int valence();
    void oneRing(Point3f *p);
    Point3f p;
    SDFace *startFace = nullptr;
    SDVertex *child = nullptr;
    bool regular = false, boundary = false;
};

struct SDFace {
    // SDFace Constructor
    SDFace() {
        for (int i = 0; i < 3; ++i) {
            v[i] = nullptr;
            f[i] = nullptr;
        }
        for (int i = 0; i < 4; ++i) {
            children[i] = nullptr;
        }
    }

    // SDFace Methods
    int vnum(SDVertex *vert) const {
        for (int i = 0; i < 3; ++i) {
            if (v[i] == vert) {
                return i;
            }
        }

        return -1;
    }
    SDFace *nextFace(SDVertex *vert) {
        return f[vnum(vert)];
    }
    SDFace *prevFace(SDVertex *vert) {
        return f[PREV(vnum(vert))];
    }
    SDVertex *nextVert(SDVertex *vert) {
        return v[NEXT(vnum(vert))];
    }
    SDVertex *prevVert(SDVertex *vert) {
        return v[PREV(vnum(vert))];
    }
    SDVertex *otherVert(SDVertex *v0, SDVertex *v1) {
        for (int i = 0; i < 3; ++i) {
            if (v[i] != v0 && v[i] != v1) {
                return v[i];
            }
        }

        return nullptr;
    }
    SDVertex *v[3];
    SDFace *f[3];
    SDFace *children[4];
};

struct SDEdge {
    // SDEdge Constructor
    SDEdge(SDVertex *v0 = nullptr, SDVertex *v1 = nullptr) {
        v[0] = std::min(v0, v1);
        v[1] = std::max(v0, v1);
        f[0] = f[1] = nullptr;
        f0edgeNum = -1;
    }

    // SDEdge Comparison Function
    bool operator<(const SDEdge &e2) const {
        if (v[0] == e2.v[0])
            return v[1] < e2.v[1];
        return v[0] < e2.v[0];
    }
    SDVertex *v[2];
    SDFace *f[2];
    int f0edgeNum;
};

// LoopSubdiv Inline Functions
inline int SDVertex::valence() {
    SDFace *f = startFace;
    if (!boundary) {
        // Compute valence of interior vertex
        int nf = 1;
        while ((f = f->nextFace(this)) != startFace) {
            ++nf;
        }
        return nf;
    } else {
        // Compute valence of boundary vertex
        int nf = 1;
        while ((f = f->nextFace(this)) != nullptr) {
            ++nf;
        }
        f = startFace;
        while ((f = f->prevFace(this)) != nullptr) {
            ++nf;
        }
        return nf + 1;
    }
}

inline FloatType beta(int valence) {
    if (valence == 3) {
        return 3.f / 16.f;
    } else {
        return 3.f / (8.f * valence);
    }
}

inline FloatType loopGamma(int valence) {
    return 1.f / (valence + 3.f / (8.f * beta(valence)));
}

static Point3f weightOneRing(SDVertex *vert, FloatType beta) {
    // Put _vert_ one-ring in _pRing_
    int valence = vert->valence();
    auto pRing = std::vector<Point3f>(valence);

    vert->oneRing(pRing.data());
    Point3f p = (1 - valence * beta) * vert->p;
    for (int i = 0; i < valence; ++i) {
        p += beta * pRing[i];
    }

    return p;
}

void SDVertex::oneRing(Point3f *p) {
    if (!boundary) {
        // Get one-ring vertices for interior vertex
        SDFace *face = startFace;
        do {
            *p++ = face->nextVert(this)->p;
            face = face->nextFace(this);
        } while (face != startFace);
    } else {
        // Get one-ring vertices for boundary vertex
        SDFace *face = startFace, *f2;
        while ((f2 = face->nextFace(this)) != nullptr)
            face = f2;
        *p++ = face->nextVert(this)->p;
        do {
            *p++ = face->prevVert(this)->p;
            face = face->prevFace(this);
        } while (face != nullptr);
    }
}

static Point3f weightBoundary(SDVertex *vert, FloatType beta) {
    // Put _vert_ one-ring in _pRing_
    int valence = vert->valence();
    auto pRing = std::vector<Point3f>(valence);

    vert->oneRing(pRing.data());
    Point3f p = (1 - 2 * beta) * vert->p;
    p += beta * pRing[0];
    p += beta * pRing[valence - 1];
    return p;
}

} // namespace

LoopSubdivide::LoopSubdivide(int nLevels, const std::vector<int> &vertexIndices,
                             const std::vector<Point3f> &p) {
    std::vector<SDVertex *> vertices;
    std::vector<SDFace *> faces;
    // Allocate _LoopSubdiv_ vertices and faces
    std::unique_ptr<SDVertex[]> verts = std::make_unique<SDVertex[]>(p.size());
    for (int i = 0; i < p.size(); ++i) {
        verts[i] = SDVertex(p[i]);
        vertices.push_back(&verts[i]);
    }

    size_t nFaces = vertexIndices.size() / 3;
    std::unique_ptr<SDFace[]> fs = std::make_unique<SDFace[]>(nFaces);
    for (int i = 0; i < nFaces; ++i) {
        faces.push_back(&fs[i]);
    }

    // Set face to vertex pointers
    const int *vp = vertexIndices.data();
    for (size_t i = 0; i < nFaces; ++i, vp += 3) {
        SDFace *f = faces[i];
        for (int j = 0; j < 3; ++j) {
            SDVertex *v = vertices[vp[j]];
            f->v[j] = v;
            v->startFace = f;
        }
    }

    // Set neighbor pointers in _faces_
    std::set<SDEdge> edges;
    for (int i = 0; i < nFaces; ++i) {
        SDFace *f = faces[i];
        for (int edgeNum = 0; edgeNum < 3; ++edgeNum) {
            // Update neighbor pointer for _edgeNum_
            int v0 = edgeNum, v1 = NEXT(edgeNum);
            SDEdge e(f->v[v0], f->v[v1]);
            if (edges.find(e) == edges.end()) {
                // Handle new edge
                e.f[0] = f;
                e.f0edgeNum = edgeNum;
                edges.insert(e);
            } else {
                // Handle previously seen edge
                e = *edges.find(e);
                e.f[0]->f[e.f0edgeNum] = f;
                f->f[edgeNum] = e.f[0];
                edges.erase(e);
            }
        }
    }

    // Finish vertex initialization
    for (size_t i = 0; i < p.size(); ++i) {
        SDVertex *v = vertices[i];
        SDFace *f = v->startFace;
        do {
            f = f->nextFace(v);
        } while ((f != nullptr) && f != v->startFace);
        v->boundary = (f == nullptr);

        if (!v->boundary && v->valence() == 6) {
            v->regular = true;
        } else if (v->boundary && v->valence() == 4) {
            v->regular = true;
        } else {
            v->regular = false;
        }
    }

    // Refine _LoopSubdiv_ into triangles
    std::vector<SDFace *> f = faces;
    std::vector<SDVertex *> v = vertices;

    for (int i = 0; i < nLevels; ++i) {
        // Update _f_ and _v_ for next level of subdivision
        std::vector<SDFace *> newFaces;
        std::vector<SDVertex *> newVertices;

        // Allocate next level of children in mesh tree
        for (SDVertex *vertex : v) {
            vertex->child = new SDVertex();
            vertex->child->regular = vertex->regular;
            vertex->child->boundary = vertex->boundary;
            newVertices.push_back(vertex->child);
        }
        for (SDFace *face : f) {
            for (int k = 0; k < 4; ++k) {
                face->children[k] = new SDFace();
                newFaces.push_back(face->children[k]);
            }
        }

        // Update vertex positions and create new edge vertices

        // Update vertex positions for even vertices
        for (SDVertex *vertex : v) {
            if (!vertex->boundary) {
                // Apply one-ring rule for even vertex
                if (vertex->regular)
                    vertex->child->p = weightOneRing(vertex, 1.f / 16.f);
                else
                    vertex->child->p = weightOneRing(vertex, beta(vertex->valence()));
            } else {
                // Apply boundary rule for even vertex
                vertex->child->p = weightBoundary(vertex, 1.f / 8.f);
            }
        }

        // Compute new odd edge vertices
        std::map<SDEdge, SDVertex *> edgeVerts;
        for (SDFace *face : f) {
            for (int k = 0; k < 3; ++k) {
                // Compute odd vertex on _k_th edge
                SDEdge edge(face->v[k], face->v[NEXT(k)]);
                SDVertex *vert = edgeVerts[edge];
                if (vert == nullptr) {
                    // Create and initialize new odd vertex
                    vert = new SDVertex();
                    newVertices.push_back(vert);
                    vert->regular = true;
                    vert->boundary = (face->f[k] == nullptr);
                    vert->startFace = face->children[3];

                    // Apply edge rules to compute new vertex position
                    if (vert->boundary) {
                        vert->p = 0.5f * edge.v[0]->p;
                        vert->p += 0.5f * edge.v[1]->p;
                    } else {
                        vert->p = 3.f / 8.f * edge.v[0]->p;
                        vert->p += 3.f / 8.f * edge.v[1]->p;
                        vert->p += 1.f / 8.f * face->otherVert(edge.v[0], edge.v[1])->p;
                        vert->p += 1.f / 8.f * face->f[k]->otherVert(edge.v[0], edge.v[1])->p;
                    }
                    edgeVerts[edge] = vert;
                }
            }
        }

        // Update new mesh topology

        // Update even vertex face pointers
        for (SDVertex *vertex : v) {
            int vertNum = vertex->startFace->vnum(vertex);
            vertex->child->startFace = vertex->startFace->children[vertNum];
        }

        // Update face neighbor pointers
        for (SDFace *face : f) {
            for (int j = 0; j < 3; ++j) {
                // Update children _f_ pointers for siblings
                face->children[3]->f[j] = face->children[NEXT(j)];
                face->children[j]->f[NEXT(j)] = face->children[3];

                // Update children _f_ pointers for neighbor children
                SDFace *f2 = face->f[j];
                face->children[j]->f[j] =
                    f2 != nullptr ? f2->children[f2->vnum(face->v[j])] : nullptr;
                f2 = face->f[PREV(j)];
                face->children[j]->f[PREV(j)] =
                    f2 != nullptr ? f2->children[f2->vnum(face->v[j])] : nullptr;
            }
        }

        // Update face vertex pointers
        for (SDFace *face : f) {
            for (int j = 0; j < 3; ++j) {
                // Update child vertex pointer to new even vertex
                face->children[j]->v[j] = face->v[j]->child;

                // Update child vertex pointer to new odd vertex
                SDVertex *vert = edgeVerts[SDEdge(face->v[j], face->v[NEXT(j)])];
                face->children[j]->v[NEXT(j)] = vert;
                face->children[NEXT(j)]->v[j] = vert;
                face->children[3]->v[j] = vert;
            }
        }

        // Prepare for next level of subdivision
        f = newFaces;
        v = newVertices;
    }

    // Push vertices to limit surface
    std::vector<Point3f> _p_limit(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        _p_limit[i] = v[i]->boundary ? weightBoundary(v[i], 1.f / 5.f)
                                     : weightOneRing(v[i], loopGamma(v[i]->valence()));
    }

    for (size_t i = 0; i < v.size(); ++i) {
        v[i]->p = _p_limit[i];
    }

    // Compute vertex tangents on limit surface

    normals.reserve(v.size());
    std::vector<Point3f> pRing(16, Point3f());
    for (SDVertex *vertex : v) {
        Vector3f S(0, 0, 0), T(0, 0, 0);
        int valence = vertex->valence();
        if (valence > (int)pRing.size()) {
            pRing.resize(valence);
        }

        vertex->oneRing(&pRing[0]);
        if (!vertex->boundary) {
            // Compute tangents of interior face
            for (int j = 0; j < valence; ++j) {
                S += std::cos(2 * compute_pi() * j / valence) * pRing[j].to_vector3();
                T += std::sin(2 * compute_pi() * j / valence) * pRing[j].to_vector3();
            }
        } else {
            // Compute tangents of boundary face
            S = pRing[valence - 1] - pRing[0];
            if (valence == 2)
                T = Vector3f(pRing[0] + pRing[1] - 2 * vertex->p);
            else if (valence == 3)
                T = pRing[1] - vertex->p;
            else if (valence == 4) // regular
                T = (-1 * pRing[0] + 2 * pRing[1] + 2 * pRing[2] + -1 * pRing[3] + -2 * vertex->p)
                        .to_vector3();
            else {
                FloatType theta = compute_pi() / FloatType(valence - 1);
                T = (std::sin(theta) * (pRing[0] + pRing[valence - 1])).to_vector3();
                for (int k = 1; k < valence - 1; ++k) {
                    FloatType wt = (2 * std::cos(theta) - 2) * std::sin((k)*theta);
                    T += (wt * pRing[k]).to_vector3();
                }
                T = -T;
            }
        }
        normals.push_back(Normal3f(S.cross(T)));
    }

    {
        size_t n_triangles = f.size();
        std::vector<int> _vertex_indices(3 * n_triangles);
        int *ptr_vertex_indices = _vertex_indices.data();
        size_t totVerts = v.size();
        std::map<SDVertex *, int> usedVerts;
        for (size_t i = 0; i < totVerts; ++i)
            usedVerts[v[i]] = i;
        for (size_t i = 0; i < n_triangles; ++i) {
            for (int j = 0; j < 3; ++j) {
                *ptr_vertex_indices = usedVerts[f[i]->v[j]];
                ++ptr_vertex_indices;
            }
        }

        vertex_indices = _vertex_indices;
        p_limit = _p_limit;
    }
}
