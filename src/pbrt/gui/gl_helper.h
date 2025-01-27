#pragma once

#include <iomanip>
#include <pbrt/euclidean_space/point2.h>
#include <pbrt/gpu/gpu_memory_allocator.h>
#include <pbrt/gui/shader.h>
#include <sstream>

class GLHelper {
    uint VBO = 0;
    uint VAO = 0;
    uint EBO = 0;
    GLFWwindow *window = nullptr;

    Shader shader;
    unsigned int texture = 0;

    bool initialized = false;

    Point2i image_resolution = Point2i(0, 0);

    GPUMemoryAllocator allocator;

  public:
    uint8_t *gpu_frame_buffer = nullptr;

    ~GLHelper() {
        if (initialized) {
            this->release();
        }
    }

    void init(const std::string &title, const Point2i &_image_resolution) {
        initialized = true;

        image_resolution = _image_resolution;
        const uint num_pixels = _image_resolution.x * _image_resolution.y;

        gpu_frame_buffer = allocator.allocate<uint8_t>(3 * num_pixels);
        for (uint idx = 0; idx < num_pixels; ++idx) {
            gpu_frame_buffer[idx * 3 + 0] = 0;
            gpu_frame_buffer[idx * 3 + 1] = 0;
            gpu_frame_buffer[idx * 3 + 2] = 0;
        }

        glfwInit();
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
        // thus disable window resizing

        const GLFWvidmode *gflw_vid_mode = glfwGetVideoMode(glfwGetPrimaryMonitor());
        const auto monitor_resolution = Point2i(gflw_vid_mode->width, gflw_vid_mode->height);

        auto window_dimension = _image_resolution;

        auto scale_numerator = 20;
        auto scale_denominator = 20;
        while (true) {
            // compute the maximal window size that can fit into the user screen
            window_dimension = _image_resolution * scale_numerator / scale_denominator;

            const auto window_ratio = 0.9;
            if (window_dimension.x <= monitor_resolution.x * window_ratio &&
                window_dimension.y <= monitor_resolution.y * window_ratio) {
                break;
            }

            scale_numerator -= 1;
            if (scale_numerator <= 0) {
                REPORT_FATAL_ERROR();
            }
        }

        create_window(window_dimension.x, window_dimension.y, title);
        glfwMakeContextCurrent(window);

        // center the window
        glfwSetWindowPos(window, (monitor_resolution.x - window_dimension.x) / 2,
                         (monitor_resolution.y - window_dimension.y) / 2);

        // glad: load all OpenGL function pointers
        // ---------------------------------------
        if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
            std::cout << "ERROR: failed to initialize GLAD\n";
            REPORT_FATAL_ERROR();
        }

        build_triangles();
    }

    void release() {
        // optional: de-allocate all resources once they've outlived their purpose:
        // ------------------------------------------------------------------------
        glDeleteVertexArrays(1, &VAO);
        glDeleteBuffers(1, &VBO);
        glDeleteBuffers(1, &EBO);

        glfwTerminate();
    }

    void create_window(uint width, uint height, const std::string &window_initial_name) {
        window = glfwCreateWindow(width, height, window_initial_name.c_str(), NULL, NULL);
        if (window == NULL) {
            std::cout << "ERROR: failed to create GLFW window" << std::endl;
            glfwTerminate();
            REPORT_FATAL_ERROR();
        }
    }

    static std::string assemble_title(const FloatType progress_percentage) {
        std::stringstream stream;
        stream << std::fixed << std::setprecision(1) << (progress_percentage * 100.0);
        return stream.str() + "%";
    }

    void draw_frame(const std::string &title) {
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image_resolution.x, image_resolution.y, 0, GL_RGB,
                     GL_UNSIGNED_BYTE, this->gpu_frame_buffer);

        glGenerateMipmap(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, texture);

        shader.use();
        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
        glfwSwapBuffers(window);

        glfwSetWindowTitle(window, title.c_str());
        glfwPollEvents();
    }

    void build_triangles() {
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
};
