#pragma once

class ParameterDict {
  public:
    ParameterDict(const std::vector<Token> &tokens) {
        // the 1st token is Keyword
        // the 2nd token is String
        // e.g. { Shape "trianglemesh" }, { Camera "perspective" }

        for (int idx = 2; idx < tokens.size(); idx += 2) {
            if (tokens[idx].type != TokenType::Variable) {
                throw std::runtime_error("expect token Variable");
            }

            auto variable_type = tokens[idx].values[0];
            auto variable_name = tokens[idx].values[1];

            if (variable_type == "integer") {
                integers[variable_name] = tokens[idx + 1].to_integers();
                continue;
            }

            if (variable_type == "float") {
                floats[variable_name] = tokens[idx + 1].to_floats();
                continue;
            }

            if (variable_type == "string") {
                strings[variable_name] = tokens[idx + 1].values[0];
                continue;
            }

            if (variable_type == "point2") {
                auto numbers = tokens[idx + 1].to_floats();
                auto p = std::vector<Point2f>(numbers.size() / 2);
                for (int k = 0; k < p.size(); k++) {
                    p[k] = Point2f(numbers[k * 2], numbers[k * 2 + 1]);
                }

                point2s[variable_name] = p;
                continue;
            }

            if (variable_type == "point3") {
                auto numbers = tokens[idx + 1].to_floats();
                auto p = std::vector<Point3f>(numbers.size() / 3);
                for (int k = 0; k < p.size(); k++) {
                    p[k] = Point3f(numbers[k * 3], numbers[k * 3 + 1], numbers[k * 3 + 2]);
                }

                point3s[variable_name] = p;
                continue;
            }

            if (variable_type == "normal") {
                auto numbers = tokens[idx + 1].to_floats();
                auto n = std::vector<Normal3f>(numbers.size() / 3);
                for (int k = 0; k < n.size(); k++) {
                    n[k] = Normal3f(numbers[k * 3], numbers[k * 3 + 1], numbers[k * 3 + 2]);
                }
                normals[variable_name] = n;
                continue;
            }

            std::cerr << "\nParameterDict(): unknown variable type: `" << variable_type << "`\n\n";
            throw std::runtime_error("unknown variable type");
        }
    }

    std::vector<int> get_integer(const std::string &key) const {
        if (integers.find(key) == integers.end()) {
            return {};
        }

        return integers.at(key);
    }

    FloatType get_float(const std::string &key, FloatType default_val) const {
        if (floats.find(key) == floats.end()) {
            return default_val;
        }

        return floats.at(key)[0];
    }

    std::string get_string(const std::string &key) const {
        return strings.at(key);
    }

    std::vector<Point2f> get_point2(const std::string &key) const {
        if (point2s.find(key) == point2s.end()) {
            return {};
        }

        return point2s.at(key);
    }

    std::vector<Point3f> get_point3(const std::string &key) const {
        if (point3s.find(key) == point3s.end()) {
            return {};
        }

        return point3s.at(key);
    }

    friend std::ostream &operator<<(std::ostream &stream, const ParameterDict &parameters) {
        if (!parameters.integers.empty()) {
            stream << "integers:\n";
            parameters.print_dict(stream, parameters.integers);
            stream << "\n";
        }

        if (!parameters.point2s.empty()) {
            stream << "Poin2f:\n";
            parameters.print_dict(stream, parameters.point2s);
            stream << "\n";
        }

        if (!parameters.point3s.empty()) {
            stream << "Poin3f:\n";
            parameters.print_dict(stream, parameters.point3s);
            stream << "\n";
        }

        return stream;
    }

  private:
    std::map<std::string, std::vector<Point2f>> point2s;
    std::map<std::string, std::vector<Point3f>> point3s;
    std::map<std::string, std::vector<Normal3f>> normals;
    std::map<std::string, std::vector<int>> integers;
    std::map<std::string, std::vector<FloatType>> floats;
    std::map<std::string, std::string> strings;

    template <typename T>
    void print_dict(std::ostream &stream,
                    const std::map<std::string, std::vector<T>> kv_map) const {
        for (const auto &kv : kv_map) {
            stream << kv.first << ": { ";
            for (const auto &x : kv.second) {
                stream << x << ", ";
            }
            stream << "}\n";
        }
    }
};
