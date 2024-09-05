#pragma once

#include "shader.h"

struct GLObject {
    uint VBO = 0;
    uint VAO = 0;
    uint EBO = 0;
    GLFWwindow *window = nullptr;

    Shader shader;
    unsigned int texture;

    void release() {
        // optional: de-allocate all resources once they've outlived their purpose:
        // ------------------------------------------------------------------------
        glDeleteVertexArrays(1, &VAO);
        glDeleteBuffers(1, &VBO);
        glDeleteBuffers(1, &EBO);
    }

    void create_window(uint width, uint height, const std::string &window_initial_name) {
        window = glfwCreateWindow(width, height, window_initial_name.c_str(), NULL, NULL);
        if (window == NULL) {
            std::cout << "ERROR: failed to create GLFW window" << std::endl;
            glfwTerminate();
            REPORT_FATAL_ERROR();
        }
    }

    void build() {
        shader.build();

        // set up vertex data (and buffer(s)) and configure vertex attributes
        // ------------------------------------------------------------------
        const float vertices[] = {
            // positions          // colors           // texture coords
            1.0f,  1.0f,  0.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f, // top right
            1.0f,  -1.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, // bottom right
            -1.0f, -1.0f, 0.0f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f, // bottom left
            -1.0f, 1.0f,  0.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f  // top left
        };

        const unsigned int indices[] = {
            0, 1, 3, // first triangle
            1, 2, 3  // second triangle
        };

        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);
        glGenBuffers(1, &EBO);

        glBindVertexArray(VAO);

        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

        // position attribute
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void *)0);
        glEnableVertexAttribArray(0);
        // color attribute
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float),
                              (void *)(3 * sizeof(float)));
        glEnableVertexAttribArray(1);
        // texture coord attribute
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float),
                              (void *)(6 * sizeof(float)));
        glEnableVertexAttribArray(2);

        // load and create a texture
        // -------------------------
        glGenTextures(1, &texture);
        glBindTexture(GL_TEXTURE_2D,
                      texture); // all upcoming GL_TEXTURE_2D operations now have effect on this
                                // texture object
        // set the texture wrapping parameters
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S,
                        GL_REPEAT); // set texture wrapping to GL_REPEAT (default wrapping method)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        // set texture filtering parameters
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    }

    void use_shader() {
        shader.use();
    }
};
