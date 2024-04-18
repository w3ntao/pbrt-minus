#pragma once

enum class TokenType {
    Illegal,
    EndOfFile,
    RightBracket,

    WorldBegin,
    AttributeBegin,
    AttributeEnd,

    Keyword,
    Number,
    String,
    Variable,
    List,
};

static std::ostream &operator<<(std::ostream &stream, const TokenType type) {
    switch (type) {
    case TokenType::Illegal: {
        stream << "Illegal";
        break;
    }
    case TokenType::EndOfFile: {
        stream << "EOF";
        break;
    }
    case TokenType::RightBracket: {
        stream << "RightBracket";
        break;
    }
    case TokenType::WorldBegin: {
        stream << "WorldBegin";
        break;
    }
    case TokenType::AttributeBegin: {
        stream << "AttributeBegin";
        break;
    }
    case TokenType::AttributeEnd: {
        stream << "AttributeEnd";
        break;
    }

    case TokenType::Keyword: {
        stream << "Keyword";
        break;
    }
    case TokenType::Number: {
        stream << "Number";
        break;
    }
    case TokenType::String: {
        stream << "String";
        break;
    }
    case TokenType::Variable: {
        stream << "Variable";
        break;
    }
    case TokenType::List: {
        stream << "List";
        break;
    }
    default: {
        throw std::runtime_error("unknown TokenType");
    }
    }

    return stream;
}

class Token {
  public:
    TokenType type;
    std::vector<std::string> values;

    explicit Token(TokenType _type) : type(_type) {}

    Token(TokenType _type, const std::string &_value) : type(_type), values({_value}) {}

    Token(TokenType _type, const std::vector<std::string> &_value) : type(_type), values(_value) {}

    bool operator==(const Token &t) const {
        if (type != t.type) {
            return false;
        }

        if (values.size() != t.values.size()) {
            return false;
        }

        for (int i = 0; i < values.size(); ++i) {
            if (values[i] != t.values[i]) {
                return false;
            }
        }

        return true;
    }

    bool operator!=(const Token &t) const {
        return !(*this == t);
    }

    FloatType to_float() const {
        if (type != TokenType::Number) {
            throw std::runtime_error("you should only invoke it with type Number.");
        }

        return stod(values[0]);
    }

    std::vector<FloatType> to_floats() const {
        std::vector<FloatType> floats(values.size());
        for (int idx = 0; idx < values.size(); idx++) {
            floats[idx] = stod(values[idx]);
        }

        return floats;
    }

    std::vector<int> to_integers() const {
        std::vector<int> integers(values.size());
        for (int idx = 0; idx < values.size(); idx++) {
            integers[idx] = stoi(values[idx]);
        }

        return integers;
    }

    friend std::ostream &operator<<(std::ostream &stream, const Token &token) {
        stream << token.type;
        if (!token.values.empty()) {
            stream << ": { ";
            for (const auto &x : token.values) {
                stream << x << ", ";
            }
            stream << "}";
        }

        return stream;
    }
};
