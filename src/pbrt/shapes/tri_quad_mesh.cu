#include "pbrt/shapes/tri_quad_mesh.h"

struct FaceCallbackContext {
    int face[4];
    std::vector<int> triIndices, quadIndices;
};

void rply_message_callback(p_ply ply, const char *message) {
    printf("rply: %s", message);
}

/* Callback to handle vertex data from RPly */
int rply_vertex_callback(p_ply_argument argument) {
    FloatType *buffer;
    long index, flags;

    ply_get_argument_user_data(argument, (void **)&buffer, &flags);
    ply_get_argument_element(argument, nullptr, &index);

    int stride = (flags & 0x0F0) >> 4;
    int offset = flags & 0x00F;

    buffer[index * stride + offset] = (float)ply_get_argument_value(argument);

    return 1;
}

/* Callback to handle face data from RPly */
int rply_face_callback(p_ply_argument argument) {
    FaceCallbackContext *context;
    long flags;
    ply_get_argument_user_data(argument, (void **)&context, &flags);

    long length, value_index;
    ply_get_argument_property(argument, nullptr, &length, &value_index);

    if (length != 3 && length != 4) {
        printf("plymesh: Ignoring face with %d vertices (only triangles and quads "
               "are supported!)",
               (int)length);
        return 1;
    } else if (value_index < 0) {
        return 1;
    }

    if (value_index >= 0) {
        context->face[value_index] = (int)ply_get_argument_value(argument);
    }

    if (value_index == length - 1) {
        if (length == 3)
            for (int i = 0; i < 3; ++i)
                context->triIndices.push_back(context->face[i]);
        else {
            // Note: modify order since we're specifying it as a blp...
            context->quadIndices.push_back(context->face[0]);
            context->quadIndices.push_back(context->face[1]);
            context->quadIndices.push_back(context->face[3]);
            context->quadIndices.push_back(context->face[2]);
        }
    }

    return 1;
}

int rply_faceindex_callback(p_ply_argument argument) {
    std::vector<int> *faceIndices;
    long flags;
    ply_get_argument_user_data(argument, (void **)&faceIndices, &flags);

    faceIndices->push_back((int)ply_get_argument_value(argument));

    return 1;
}

TriQuadMesh TriQuadMesh::read_ply(const std::string &filename) {
    TriQuadMesh mesh;

    p_ply ply = ply_open(filename.c_str(), rply_message_callback, 0, nullptr);
    if (!ply) {
        printf("Couldn't open PLY file \"%s\"", filename.c_str());
        exit(1);
    }

    if (ply_read_header(ply) == 0) {
        printf("Unable to read the header of PLY file \"%s\"", filename.c_str());
        exit(1);
    }

    p_ply_element element = nullptr;
    size_t vertexCount = 0;
    size_t faceCount = 0;

    /* Inspect the structure of the PLY file */
    while ((element = ply_get_next_element(ply, element)) != nullptr) {
        const char *name;
        long nInstances;

        ply_get_element_info(element, &name, &nInstances);
        if (strcmp(name, "vertex") == 0) {
            vertexCount = nInstances;
        } else if (strcmp(name, "face") == 0) {
            faceCount = nInstances;
        }
    }

    if (vertexCount == 0 || faceCount == 0) {
        printf("%s: PLY file is invalid! No face/vertex elements found!", filename.c_str());
        exit(1);
    }

    mesh.p.resize(vertexCount);
    if (ply_set_read_cb(ply, "vertex", "x", rply_vertex_callback, mesh.p.data(), 0x30) == 0 ||
        ply_set_read_cb(ply, "vertex", "y", rply_vertex_callback, mesh.p.data(), 0x31) == 0 ||
        ply_set_read_cb(ply, "vertex", "z", rply_vertex_callback, mesh.p.data(), 0x32) == 0) {
        printf("%s: Vertex coordinate property not found!", filename.c_str());
        exit(1);
    }

    mesh.n.resize(vertexCount);
    if (ply_set_read_cb(ply, "vertex", "nx", rply_vertex_callback, mesh.n.data(), 0x30) == 0 ||
        ply_set_read_cb(ply, "vertex", "ny", rply_vertex_callback, mesh.n.data(), 0x31) == 0 ||
        ply_set_read_cb(ply, "vertex", "nz", rply_vertex_callback, mesh.n.data(), 0x32) == 0) {
        mesh.n.resize(0);
    }

    /* There seem to be lots of different conventions regarding UV coordinate
     * names */
    mesh.uv.resize(vertexCount);
    if (((ply_set_read_cb(ply, "vertex", "u", rply_vertex_callback, mesh.uv.data(), 0x20) != 0) &&
         (ply_set_read_cb(ply, "vertex", "v", rply_vertex_callback, mesh.uv.data(), 0x21) != 0)) ||
        ((ply_set_read_cb(ply, "vertex", "s", rply_vertex_callback, mesh.uv.data(), 0x20) != 0) &&
         (ply_set_read_cb(ply, "vertex", "t", rply_vertex_callback, mesh.uv.data(), 0x21) != 0)) ||
        ((ply_set_read_cb(ply, "vertex", "texture_u", rply_vertex_callback, mesh.uv.data(), 0x20) !=
          0) &&
         (ply_set_read_cb(ply, "vertex", "texture_v", rply_vertex_callback, mesh.uv.data(), 0x21) !=
          0)) ||
        ((ply_set_read_cb(ply, "vertex", "texture_s", rply_vertex_callback, mesh.uv.data(), 0x20) !=
          0) &&
         (ply_set_read_cb(ply, "vertex", "texture_t", rply_vertex_callback, mesh.uv.data(), 0x21) !=
          0))) {
        // do nothing
    } else {
        mesh.uv.resize(0);
    }

    FaceCallbackContext context;
    context.triIndices.reserve(faceCount * 3);
    context.quadIndices.reserve(faceCount * 4);
    if (ply_set_read_cb(ply, "face", "vertex_indices", rply_face_callback, &context, 0) == 0) {
        printf("%s: vertex indices not found in PLY file", filename.c_str());
        exit(1);
    }

    if (ply_set_read_cb(ply, "face", "face_indices", rply_faceindex_callback, &mesh.faceIndices,
                        0) != 0) {
        mesh.faceIndices.reserve(faceCount);
    }

    if (ply_read(ply) == 0) {
        printf("%s: unable to read the contents of PLY file", filename.c_str());
        exit(1);
    }

    mesh.triIndices = std::move(context.triIndices);
    mesh.quadIndices = std::move(context.quadIndices);

    ply_close(ply);

    for (int idx : mesh.triIndices) {
        if (idx < 0 || idx >= mesh.p.size()) {
            printf("plymesh: Vertex index %d is out of bounds! "
                   "Valid range is [0..%d)",
                   idx, int(mesh.p.size()));
            exit(1);
        }
    }

    for (int idx : mesh.quadIndices) {
        if (idx < 0 || idx >= mesh.p.size()) {
            printf("plymesh: Vertex index %d is out of bounds! "
                   "Valid range is [0..%d)",
                   idx, int(mesh.p.size()));
            exit(1);
        }
    }

    return mesh;
}