#pragma once

#include "pbrt/scene/tokenizer.h"
#include <fstream>
#include <iterator>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

class Lexer {

  private:
    bool string_starts_with(const std::string &full_str, const std::string &beginning) {
        return full_str.rfind(beginning, 0) == 0;
    }

    bool is_letter(char ch) {
        return (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || ch == '_';
    }

    bool is_digit(char ch) {
        return ch >= '0' && ch <= '9';
    }

    Token parse_identifier(const std::string &identifier) {
        if (identifier == "WorldBegin") {
            return Token(TokenType::WorldBegin);
        }

        if (identifier == "AttributeBegin") {
            return Token(TokenType::AttributeBegin);
        }

        if (identifier == "AttributeEnd") {
            return Token(TokenType::AttributeEnd);
        }

        if (identifier == "ObjectBegin") {
            skip_space();
            auto object_name = read_next_quoted_string();
            return Token(TokenType::ObjectBegin, {object_name});
        }

        if (identifier == "ObjectEnd") {
            return Token(TokenType::ObjectEnd);
        }

        if (identifier == "ObjectInstance") {
            skip_space();
            auto object_name = read_next_quoted_string();
            return Token(TokenType::ObjectInstance, {object_name});
        }

        if (identifier == "true" || identifier == "false") {
            return Token(TokenType::String, identifier);
        }

        return Token(TokenType::Keyword, identifier);
    }

  public:
    std::string input;
    int position;
    std::optional<char> current_char;
    int line_number;
    bool in_bracket = false;

    Lexer(const std::string &filename)
        : position(0), current_char(std::nullopt), line_number(1), in_bracket(false) {
        std::ifstream file_stream(filename);
        std::stringstream buffer;
        buffer << file_stream.rdbuf();
        input = buffer.str();

        if (input.size() == 0) {
            throw std::runtime_error("fail to build Lexer");
        }
    }

    void read_char() {
        if (position >= input.size()) {
            current_char = std::nullopt;
        } else {
            current_char = input[position];
        }

        position += 1;
    }

    std::string read_number() {
        int last_position = position - 1;

        while (true) {
            if (!current_char) {
                break;
            }

            if (current_char == '-' || current_char == 'e' || current_char == '.' ||
                is_digit(current_char.value())) {
                // `e` for scientific notation
                read_char();
                continue;
            }

            break;
        }

        return input.substr(last_position, position - 1 - last_position);
    }

    std::vector<std::string> read_list() {
        read_char(); // consume '['
        std::vector<std::string> values;
        while (true) {
            auto token = next_token();

            if (token.type == TokenType::RightBracket) {
                break;
            }

            if (token.type == TokenType::Number || token.type == TokenType::String) {
                values.push_back(token.values[0]);
                continue;
            }

            std::cout << "\n" << __func__ << "(): error token type: " << token.type << "\n";
            REPORT_FATAL_ERROR();
        }

        return values;
    }

    std::string read_next_quoted_string() {
        // you could get String or Variable from this token

        int last_position = position - 1;
        read_char(); // consume first quote

        while (current_char != '"') {
            read_char();
        }
        read_char(); // consume the last quote

        return input.substr(last_position + 1, position - 2 - (last_position + 1));
    }

    std::string read_identifier() {
        int last_position = position - 1;
        while (current_char.has_value() && is_letter(current_char.value())) {
            read_char();
        }

        return input.substr(last_position, (position - 1) - last_position);
    }

    void skip_space() {
        for (;;) {
            if (!current_char) {
                return;
            }

            if (current_char == ' ' || current_char == '\t' || current_char == '\n' ||
                current_char == '\r') {
                if (current_char == '\n') {
                    line_number += 1;
                }
                read_char();
                continue;
            }

            return;
        }
    }

    void skip_comment() {
        for (;;) {
            read_char();
            if (current_char == '\n') {
                line_number += 1;
                return;
            }
        }
    }

    Token next_token() {
        skip_space();
        while (current_char == '#') {
            skip_comment();
            skip_space();
        }

        if (!current_char) {
            return Token(TokenType::EndOfFile);
        }

        switch (current_char.value()) {
        case '[': {
            in_bracket = true;
            auto token = Token(TokenType::List, read_list());
            read_char();
            return token;
        }

        case ']': {
            in_bracket = false;
            read_char();
            return Token(TokenType::RightBracket);
        }

        case '"': {
            const auto variable_definitions = {
                "bool ",   "blackbody ", "float ",  "integer ",  "normal ",  "rgb ",
                "point2 ", "point3 ",    "string ", "spectrum ", "texture ", "vector3 ",
            };

            auto is_string = true;
            auto string_without_quote = read_next_quoted_string();

            if (in_bracket || string_without_quote.find(' ') == std::string::npos) {
                is_string = true;
            } else {
                for (const auto type : variable_definitions) {
                    if (string_starts_with(string_without_quote, type)) {
                        is_string = false;
                        break;
                    }
                }
            }

            if (is_string) {
                return Token(TokenType::String, string_without_quote);
            }

            // otherwise it's a Variable

            std::istringstream iss(string_without_quote);

            std::vector<std::string> split_strings;
            copy(std::istream_iterator<std::string>(iss), std::istream_iterator<std::string>(),
                 back_inserter(split_strings));

            return Token(TokenType::Variable, split_strings);
        }
        }

        if (is_letter(current_char.value())) {
            return parse_identifier(read_identifier());
        }

        if (current_char == '-' || current_char == '.' || is_digit(current_char.value())) {
            return Token(TokenType::Number, read_number());
        }

        printf("line %d: illegal char: `%c`", line_number, current_char.value());

        return Token(TokenType::Illegal);
    }
};
