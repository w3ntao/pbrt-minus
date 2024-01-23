#pragma once

#include <optional>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <iterator>
#include <algorithm>

#include "pbrt/scene/tokenizer.h"

class Lexer {

  private:
    bool is_letter(char ch) {
        return (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || ch == '_';
    }

    bool is_digit(char ch) {
        return ch >= '0' && ch <= '9';
    }

    Token parse_identifier(const std::string &identifier) {
        if (identifier == "WorldBegin") {
            return Token(WorldBegin);
        }

        if (identifier == "AttributeBegin") {
            return Token(AttributeBegin);
        }

        if (identifier == "AttributeEnd") {
            return Token(AttributeEnd);
        }

        if (identifier == "true" || identifier == "false") {
            return Token(String, identifier);
        }

        return Token(Keyword, identifier);
    }

  public:
    std::string input;
    int position;
    std::optional<char> current_char;
    int line_number;

    Lexer(const std::string &filename) : position(0), current_char(std::nullopt), line_number(1) {
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

        for (;;) {
            if (!current_char) {
                break;
            }
            if (current_char == '-' || current_char == '.' || is_digit(current_char.value())) {
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
        for (;;) {
            auto token = next_token();

            if (token.type == RightBracket) {
                break;
            }

            if (token.type == Number || token.type == String) {
                values.push_back(token.value[0]);
                continue;
            }

            throw std::runtime_error("Lexer::read_list() error");
        }

        return values;
    }

    Token read_quoted_string() {
        int last_position = position - 1;
        read_char(); // consume first quote

        while (current_char != '"') {
            read_char();
        }
        read_char(); // consume the last quote

        std::string string_without_quote =
            input.substr(last_position + 1, position - 1 - (last_position + 1));

        if (string_without_quote.find(' ') == std::string::npos) {
            return Token(String, string_without_quote);
        }

        std::istringstream iss(string_without_quote);

        std::vector<std::string> split_strings;
        copy(std::istream_iterator<std::string>(iss), std::istream_iterator<std::string>(),
             back_inserter(split_strings));

        return Token(Variable, split_strings);
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
            return Token(EndOfFile);
        }

        switch (current_char.value()) {
        case '[': {
            auto token = Token(List, read_list());
            read_char();
            return token;
        }

        case ']': {
            read_char();
            return Token(RightBracket);
        }

        case '"': {
            return read_quoted_string();
        }

        default: {
            if (is_letter(current_char.value())) {
                return parse_identifier(read_identifier());
            }

            if (current_char == '-' || is_digit(current_char.value())) {
                return Token(Number, read_number());
            }

            printf("line %d: illegal char: `%c`", line_number, current_char.value());

            return Token(Illegal);
        }
        }
    }
};
