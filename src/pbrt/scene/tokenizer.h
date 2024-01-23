#pragma once

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
};
