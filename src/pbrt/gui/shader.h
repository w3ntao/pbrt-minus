#pragma once

// clang-format off
#include <glad/glad.h>
#include <GLFW/glfw3.h>
// clang-format on

#include <iostream>
#include <string>

class Shader {
  public:
    Shader() : ID(0) {}

    // constructor generates the shader on the fly
    // ------------------------------------------------------------------------
    void build() {
        std::string vertex_code_str = "#version 330 core\n"
                                      "layout (location = 0) in vec3 aPos;\n"
                                      "layout (location = 1) in vec3 aColor;\n"
                                      "layout (location = 2) in vec2 aTexCoord;\n"
                                      "\n"
                                      "out vec3 ourColor;\n"
                                      "out vec2 TexCoord;\n"
                                      "\n"
                                      "void main()\n"
                                      "{\n"
                                      "    gl_Position = vec4(aPos, 1.0);\n"
                                      "    ourColor = aColor;\n"
                                      "    TexCoord = vec2(aTexCoord.x, aTexCoord.y);\n"
                                      "}\n";

        std::string fragment_code_str = "#version 330 core\n"
                                        "out vec4 FragColor;\n"
                                        "\n"
                                        "in vec3 ourColor;\n"
                                        "in vec2 TexCoord;\n"
                                        "\n"
                                        "// texture sampler\n"
                                        "uniform sampler2D texture1;\n"
                                        "\n"
                                        "void main()\n"
                                        "{\n"
                                        "    FragColor = texture(texture1, TexCoord);\n"
                                        "}\n";

        const char *vShaderCode = vertex_code_str.c_str();
        const char *fShaderCode = fragment_code_str.c_str();
        // 2. compile shaders
        unsigned int vertex, fragment;
        // vertex shader
        vertex = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vertex, 1, &vShaderCode, NULL);
        glCompileShader(vertex);
        checkCompileErrors(vertex, "VERTEX");
        // fragment Shader
        fragment = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fragment, 1, &fShaderCode, NULL);
        glCompileShader(fragment);
        checkCompileErrors(fragment, "FRAGMENT");
        // shader Program
        ID = glCreateProgram();
        glAttachShader(ID, vertex);
        glAttachShader(ID, fragment);
        glLinkProgram(ID);
        checkCompileErrors(ID, "PROGRAM");
        // delete the shaders as they're linked into our program now and no longer necessary
        glDeleteShader(vertex);
        glDeleteShader(fragment);
    }

    // activate the shader
    // ------------------------------------------------------------------------
    void use() {
        glUseProgram(ID);
    }

  private:
    unsigned int ID;

    // utility function for checking shader compilation/linking errors.
    // ------------------------------------------------------------------------
    void checkCompileErrors(unsigned int shader, std::string type) {
        int success;
        char infoLog[1024];
        if (type != "PROGRAM") {
            glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
            if (!success) {
                glGetShaderInfoLog(shader, 1024, NULL, infoLog);
                std::cout << "ERROR::SHADER_COMPILATION_ERROR of type: " << type << "\n"
                          << infoLog
                          << "\n -- --------------------------------------------------- -- "
                          << std::endl;
            }
        } else {
            glGetProgramiv(shader, GL_LINK_STATUS, &success);
            if (!success) {
                glGetProgramInfoLog(shader, 1024, NULL, infoLog);
                std::cout << "ERROR::PROGRAM_LINKING_ERROR of type: " << type << "\n"
                          << infoLog
                          << "\n -- --------------------------------------------------- -- "
                          << std::endl;
            }
        }
    }
};
