#pragma once

#include <iostream>

#include "pbrt/scene/lexer.h"

std::vector<Token> parse_pbrt_into_token(const std::string &filename) {
    auto lexer = Lexer(filename);

    std::vector<Token> tokens;
    lexer.read_char();
    for (;;) {
        auto token = lexer.next_token();

        if (token.type == Illegal) {
            printf("parsing `%s` fails at line %d\n", filename, lexer.line_number);
            throw std::runtime_error("parse_pbrt_into_token() fails");
        }

        if (token.type == EndOfFile) {
            break;
        }

        tokens.push_back(token);
    }

    return tokens;
}
