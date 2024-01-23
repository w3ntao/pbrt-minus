#pragma once

#include <stack>
#include <map>

#include "pbrt/scene/parser.h"
#include "pbrt/euclidean_space/transform.h"

class GraphicsState {
  public:
    GraphicsState() : current_transform(Transform::identity()), reversed_orientation(false) {}

    Transform current_transform;
    bool reversed_orientation;
};

class SceneBuilder {
  private:
    GraphicsState graphics_state;
    std::stack<GraphicsState> pushed_graphics_state;
    std::map<std::string, Transform> named_coordinate_systems;

    SceneBuilder() = default;

    std::vector<int> split_tokens_into_statements(const std::vector<Token> &tokens) {
        std::vector<int> keyword_range;
        for (int idx = 0; idx < tokens.size(); ++idx) {
            const auto &token = tokens[idx];
            if (token.type == WorldBegin || token.type == AttributeBegin ||
                token.type == AttributeEnd || token.type == Keyword) {
                keyword_range.push_back(idx);
            }
        }
        keyword_range.push_back(tokens.size());

        return keyword_range;
    }

    void option_lookat(const std::vector<Token> &tokens) {
        std::vector<double> data;
        for (int idx = 1; idx < tokens.size(); idx++) {
            data.push_back(tokens[idx].to_number());
        }

        auto position = Point3f(data[0], data[1], data[2]);
        auto look = Point3f(data[3], data[4], data[5]);
        auto up = Vector3f(data[6], data[7], data[8]);

        auto transform_look_at = Transform::lookat(position, look, up);

        graphics_state.current_transform *= transform_look_at;
    }

    void parse_statement(const std::vector<Token> &statements) {
        const Token &first_token = statements[0];

        switch (first_token.type) {
        case AttributeBegin: {
            pushed_graphics_state.push(graphics_state);
            return;
        }
        case AttributeEnd: {
            graphics_state = pushed_graphics_state.top();
            pushed_graphics_state.pop();
            return;
        }
        case WorldBegin: {
            graphics_state.current_transform = Transform::identity();
            named_coordinate_systems["world"] = graphics_state.current_transform;
            return;
        }
        case Keyword: {
            const auto keyword = first_token.value[0];
            if (keyword == "Camera" || keyword == "Film" || keyword == "Sampler" ||
                keyword == "Integrator" || keyword == "Material" || keyword == "AreaLightSource") {
                std::cout << "parse_statement::Keyword `" << keyword << "` ignored\n";
                return;
            }

            if (keyword == "LookAt") {
                option_lookat(statements);
                return;
            }

            std::cout << "parse_statement::Keyword `" << keyword << "` not implemented\n";

            return;
        }

        // TODO: progress 2024/01/23 parsing PBRT file
        default: {
            printf("unkown token type: `%d`\n", first_token.type);
            throw std::runtime_error("parse_statement() fail");
        }
        }
    }

  public:
    static SceneBuilder read_pbrt(const std::string &filename) {
        auto tokens = parse_pbrt_into_token(filename);

        auto builder = SceneBuilder();
        auto range_of_statements = builder.split_tokens_into_statements(tokens);

        for (int range_idx = 0; range_idx < range_of_statements.size() - 1; ++range_idx) {
            auto statements = std::vector(tokens.begin() + range_of_statements[range_idx],
                                          tokens.begin() + range_of_statements[range_idx + 1]);

            /*
            for (const auto &t : statements) {
                t.print();
            }
            std::cout << "\n\n";
            */

            builder.parse_statement(statements);
        }

        return builder;
    }
};
