#pragma once

// TODO: rewrite TypeOfToken with enum class
enum TypeOfToken {
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

std::string parsing_type_to_string(TypeOfToken type) {
    switch (type) {
    case Illegal:
        return "Illegal";
    case EndOfFile:
        return "EOF";
    case RightBracket:
        return "RightBracket";
    case WorldBegin:
        return "WorldBegin";
    case AttributeBegin:
        return "AttributeBegin";
    case AttributeEnd:
        return "AttributeEnd";

    case Keyword:
        return "Keyword";
    case Number:
        return "Number";
    case String:
        return "String";
    case Variable:
        return "Variable";
    case List:
        return "List";
    }

    throw std::runtime_error("parsing_type_to_string() error");
}

class Token {
  public:
    TypeOfToken type;
    std::vector<std::string> value;

    Token(TypeOfToken _type) : type(_type) {}

    Token(TypeOfToken _type, const std::string &_value) : type(_type), value({_value}) {}

    Token(TypeOfToken _type, const std::vector<std::string> _value) : type(_type), value(_value) {}

    bool operator==(const Token &t) const {
        if (type != t.type) {
            return false;
        }

        if (value.size() != t.value.size()) {
            return false;
        }

        for (int i = 0; i < value.size(); ++i) {
            if (value[i] != t.value[i]) {
                return false;
            }
        }

        return true;
    }

    bool operator!=(const Token &t) const {
        return !(*this == t);
    }

    double to_number() const {
        if (type != Number) {
            throw std::runtime_error("you should only invoke it with type Number.");
        }

        return stod(value[0]);
    }

    std::vector<double> to_floats() const {
        std::vector<double> floats(value.size());
        for (int idx = 0; idx < value.size(); idx++) {
            floats[idx] = stod(value[idx]);
        }

        return floats;
    }

    std::vector<int> to_integers() const {
        std::vector<int> integers(value.size());
        for (int idx = 0; idx < value.size(); idx++) {
            integers[idx] = stoi(value[idx]);
        }

        return integers;
    }

    friend std::ostream &operator<<(std::ostream &stream, const Token &token) {
        stream << parsing_type_to_string(token.type);
        if (token.value.size() > 0) {
            stream << ": { ";
            for (const auto &x : token.value) {
                stream << x << ", ";
            }
            stream << "}";
        }

        return stream;
    }
};
