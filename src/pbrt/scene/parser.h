#pragma once

#include <iostream>

#include "pbrt/scene/lexer.h"

std::vector<Token> parse_pbrt_into_token(const std::string &filename) {
    auto lexer = Lexer(filename);

    std::vector<Token> tokens;
    lexer.read_char();
    while (true) {
        auto token = lexer.next_token();

        if (token.type == TokenType::Illegal) {
            printf("parsing `%s` fails at line %d\n", filename.c_str(), lexer.line_number);
            REPORT_FATAL_ERROR();
        }

        if (token.type == TokenType::EndOfFile) {
            break;
        }

        tokens.push_back(token);
    }

    return tokens;
}
