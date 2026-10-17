#include "dlc/ir/Parser.h"
#include "dlc/ir/Builder.h"
#include "dlc/ir/Error.h"
#include "dlc/ir/OpRegistry.h"
#include <cctype>
#include <charconv>
#include <limits>
#include <map>
#include <unordered_map>

namespace dlc {

namespace {

enum class Tok {
    Eof, Ident, ValueId, FuncId, Int, Float, String,
    LParen, RParen, LBrace, RBrace, LBracket, RBracket, Comma, Colon, Equal, Arrow
};

struct Token {
    Tok kind{Tok::Eof};
    std::string text;
    int line{1};
    int col{1};
};

std::string describe(const Token& tok) {
    switch (tok.kind) {
        case Tok::Eof: return "end of input";
        case Tok::ValueId: return "'%" + tok.text + "'";
        case Tok::FuncId: return "'@" + tok.text + "'";
        case Tok::String: return "string \"" + tok.text + "\"";
        default: return "'" + tok.text + "'";
    }
}

IRError errorAt(const Token& tok, const std::string& message) {
    return IRError(std::to_string(tok.line) + ":" + std::to_string(tok.col) + ": " + message);
}

bool isNameChar(char c) {
    return std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '.' || c == '$';
}

class Lexer {
public:
    explicit Lexer(std::string_view source) : src_(source) {}

    Token next() {
        skipTrivia();
        Token tok;
        tok.line = line_;
        tok.col = col_;
        if (pos_ >= src_.size()) return tok;

        char c = src_[pos_];
        auto punct = [&](Tok kind) {
            tok.kind = kind;
            tok.text = std::string(1, c);
            advance();
            return tok;
        };
        switch (c) {
            case '(': return punct(Tok::LParen);
            case ')': return punct(Tok::RParen);
            case '{': return punct(Tok::LBrace);
            case '}': return punct(Tok::RBrace);
            case '[': return punct(Tok::LBracket);
            case ']': return punct(Tok::RBracket);
            case ',': return punct(Tok::Comma);
            case ':': return punct(Tok::Colon);
            case '=': return punct(Tok::Equal);
            case '%':
            case '@': {
                advance();
                size_t start = pos_;
                while (pos_ < src_.size() && isNameChar(src_[pos_])) advance();
                if (pos_ == start) throw errorAt(tok, std::string("expected a name after '") + c + "'");
                tok.kind = c == '%' ? Tok::ValueId : Tok::FuncId;
                tok.text = std::string(src_.substr(start, pos_ - start));
                return tok;
            }
            case '"': return lexString(tok);
            case '-':
                if (peek(1) == '>') {
                    advance();
                    advance();
                    tok.kind = Tok::Arrow;
                    tok.text = "->";
                    return tok;
                }
                return lexNumber(tok);
            default: break;
        }
        if (std::isdigit(static_cast<unsigned char>(c))) return lexNumber(tok);
        if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
            size_t start = pos_;
            while (pos_ < src_.size() && (std::isalnum(static_cast<unsigned char>(src_[pos_])) ||
                                          src_[pos_] == '_' || src_[pos_] == '.')) {
                advance();
            }
            tok.text = std::string(src_.substr(start, pos_ - start));
            tok.kind = (tok.text == "nan" || tok.text == "inf") ? Tok::Float : Tok::Ident;
            return tok;
        }
        throw errorAt(tok, std::string("unexpected character '") + c + "'");
    }

private:
    char peek(size_t ahead) const { return pos_ + ahead < src_.size() ? src_[pos_ + ahead] : '\0'; }

    void advance() {
        if (src_[pos_] == '\n') {
            ++line_;
            col_ = 1;
        } else {
            ++col_;
        }
        ++pos_;
    }

    void skipTrivia() {
        while (pos_ < src_.size()) {
            if (std::isspace(static_cast<unsigned char>(src_[pos_]))) {
                advance();
            } else if (src_[pos_] == '/' && peek(1) == '/') {
                while (pos_ < src_.size() && src_[pos_] != '\n') advance();
            } else {
                break;
            }
        }
    }

    Token lexNumber(Token tok) {
        size_t start = pos_;
        if (src_[pos_] == '-') advance();
        if (src_.substr(pos_, 3) == "inf") {
            for (int i = 0; i < 3; ++i) advance();
            tok.kind = Tok::Float;
            tok.text = "-inf";
            return tok;
        }
        if (pos_ >= src_.size() || !std::isdigit(static_cast<unsigned char>(src_[pos_]))) {
            throw errorAt(tok, "expected a number after '-'");
        }
        bool isFloat = false;
        while (pos_ < src_.size() && std::isdigit(static_cast<unsigned char>(src_[pos_]))) advance();
        if (pos_ < src_.size() && src_[pos_] == '.') {
            isFloat = true;
            advance();
            while (pos_ < src_.size() && std::isdigit(static_cast<unsigned char>(src_[pos_]))) advance();
        }
        if (pos_ < src_.size() && (src_[pos_] == 'e' || src_[pos_] == 'E')) {
            isFloat = true;
            advance();
            if (pos_ < src_.size() && (src_[pos_] == '+' || src_[pos_] == '-')) advance();
            if (pos_ >= src_.size() || !std::isdigit(static_cast<unsigned char>(src_[pos_]))) {
                throw errorAt(tok, "malformed exponent in number");
            }
            while (pos_ < src_.size() && std::isdigit(static_cast<unsigned char>(src_[pos_]))) advance();
        }
        tok.kind = isFloat ? Tok::Float : Tok::Int;
        tok.text = std::string(src_.substr(start, pos_ - start));
        return tok;
    }

    Token lexString(Token tok) {
        advance(); // opening quote
        std::string text;
        while (true) {
            if (pos_ >= src_.size() || src_[pos_] == '\n') throw errorAt(tok, "unterminated string");
            char c = src_[pos_];
            advance();
            if (c == '"') break;
            if (c == '\\') {
                if (pos_ >= src_.size()) throw errorAt(tok, "unterminated string");
                char escaped = src_[pos_];
                advance();
                switch (escaped) {
                    case 'n': text += '\n'; break;
                    case 't': text += '\t'; break;
                    case '"': text += '"'; break;
                    case '\\': text += '\\'; break;
                    default: throw errorAt(tok, std::string("unknown escape '\\") + escaped + "' in string");
                }
            } else {
                text += c;
            }
        }
        tok.kind = Tok::String;
        tok.text = std::move(text);
        return tok;
    }

    std::string_view src_;
    size_t pos_{0};
    int line_{1};
    int col_{1};
};

class Parser {
public:
    explicit Parser(std::string_view source) : lexer_(source) { tok_ = lexer_.next(); }

    ModulePtr parseModule() {
        auto module = std::make_shared<Module>("parsed_module");
        while (tok_.kind != Tok::Eof) {
            if (tok_.kind != Tok::Ident || tok_.text != "func") {
                throw errorAt(tok_, "expected 'func', got " + describe(tok_));
            }
            module->addFunction(parseFunction(*module));
        }
        return module;
    }

private:
    using SymbolTable = std::unordered_map<std::string, ValuePtr>;

    struct ParsedAttr {
        AttributeValue value;
        Token tok;
    };

    Token consume() {
        Token tok = tok_;
        tok_ = lexer_.next();
        return tok;
    }

    bool accept(Tok kind) {
        if (tok_.kind != kind) return false;
        consume();
        return true;
    }

    Token expect(Tok kind, const std::string& what) {
        if (tok_.kind != kind) throw errorAt(tok_, "expected " + what + ", got " + describe(tok_));
        return consume();
    }

    static int64_t toInt(const Token& tok) {
        int64_t value = 0;
        const char* begin = tok.text.data();
        const char* end = begin + tok.text.size();
        auto [ptr, ec] = std::from_chars(begin, end, value);
        if (ec != std::errc() || ptr != end) throw errorAt(tok, "integer " + tok.text + " is out of range");
        return value;
    }

    static double toDouble(const Token& tok) {
        if (tok.text == "nan") return std::numeric_limits<double>::quiet_NaN();
        if (tok.text == "inf") return std::numeric_limits<double>::infinity();
        if (tok.text == "-inf") return -std::numeric_limits<double>::infinity();
        double value = 0.0;
        const char* begin = tok.text.data();
        const char* end = begin + tok.text.size();
        auto [ptr, ec] = std::from_chars(begin, end, value);
        if (ec != std::errc() || ptr != end) throw errorAt(tok, "invalid number " + tok.text);
        return value;
    }

    TypePtr parseType() {
        Token name = expect(Tok::Ident, "a type such as 'f32[2,3]'");
        DataType dataType;
        if (name.text == "f32") dataType = DataType::Float32;
        else if (name.text == "i32") dataType = DataType::Int32;
        else throw errorAt(name, "unknown element type '" + name.text + "'");

        expect(Tok::LBracket, "'['");
        Shape shape;
        if (!accept(Tok::RBracket)) {
            do {
                Token dim = expect(Tok::Int, "a dimension");
                int64_t value = toInt(dim);
                if (value < 0) throw errorAt(dim, "dimensions must be non-negative");
                shape.push_back(value);
                if (checkedNumElements(shape) < 0) throw errorAt(dim, "tensor type has too many elements");
            } while (accept(Tok::Comma));
            expect(Tok::RBracket, "']'");
        }
        return std::make_shared<TensorType>(dataType, shape);
    }

    AttributeValue parseAttrValue() {
        if (tok_.kind == Tok::Int) return toInt(consume());
        if (tok_.kind == Tok::Float) return toDouble(consume());
        if (tok_.kind == Tok::String) return consume().text;
        if (tok_.kind == Tok::Ident && (tok_.text == "true" || tok_.text == "false")) {
            return int64_t{consume().text == "true" ? 1 : 0};
        }
        if (accept(Tok::LBracket)) {
            std::vector<int64_t> ints;
            std::vector<double> doubles;
            bool isFloat = false;
            if (!accept(Tok::RBracket)) {
                do {
                    if (tok_.kind == Tok::Int) {
                        int64_t value = toInt(consume());
                        ints.push_back(value);
                        doubles.push_back(static_cast<double>(value));
                    } else if (tok_.kind == Tok::Float) {
                        isFloat = true;
                        doubles.push_back(toDouble(consume()));
                    } else {
                        throw errorAt(tok_, "expected a number in the list, got " + describe(tok_));
                    }
                } while (accept(Tok::Comma));
                expect(Tok::RBracket, "']'");
            }
            if (isFloat) return doubles;
            return ints;
        }
        throw errorAt(tok_, "expected an attribute value, got " + describe(tok_));
    }

    std::map<std::string, ParsedAttr> parseAttrDict() {
        std::map<std::string, ParsedAttr> attrs;
        expect(Tok::LBrace, "'{'");
        if (accept(Tok::RBrace)) return attrs;
        do {
            Token key = expect(Tok::Ident, "an attribute name");
            expect(Tok::Equal, "'='");
            AttributeValue value = parseAttrValue();
            if (!attrs.emplace(key.text, ParsedAttr{std::move(value), key}).second) {
                throw errorAt(key, "duplicate attribute '" + key.text + "'");
            }
        } while (accept(Tok::Comma));
        expect(Tok::RBrace, "'}'");
        return attrs;
    }

    // Checks a parsed value against the attribute's declared kind (an integer literal is accepted as a float).
    static AttributeValue coerce(const ParsedAttr& attr, const AttrSpec& spec, const std::string& opName) {
        if (spec.kind == AttrKind::Double) {
            if (const int64_t* value = std::get_if<int64_t>(&attr.value)) return static_cast<double>(*value);
        }
        if (!attrHasKind(attr.value, spec.kind)) {
            throw errorAt(attr.tok, "attribute '" + spec.name + "' of '" + opName + "' must be " +
                                        attrKindName(spec.kind));
        }
        return attr.value;
    }

    ValuePtr lookup(const Token& tok, const SymbolTable& symbols) {
        auto it = symbols.find(tok.text);
        if (it == symbols.end()) throw errorAt(tok, "use of undefined value '%" + tok.text + "'");
        return it->second;
    }

    std::vector<ValuePtr> parseValueList(const SymbolTable& symbols, const std::string& what) {
        std::vector<ValuePtr> values;
        do {
            values.push_back(lookup(expect(Tok::ValueId, what), symbols));
        } while (accept(Tok::Comma));
        return values;
    }

    FunctionPtr parseFunction(const Module& module) {
        consume(); // 'func'
        Token name = expect(Tok::FuncId, "a function name such as '@main'");
        if (module.getFunction(name.text)) throw errorAt(name, "redefinition of function '@" + name.text + "'");

        struct ArgDecl {
            Token tok;
            TypePtr type;
            ArgRole role;
        };
        std::vector<ArgDecl> args;
        expect(Tok::LParen, "'('");
        if (!accept(Tok::RParen)) {
            do {
                Token id = expect(Tok::ValueId, "an argument such as '%x'");
                expect(Tok::Colon, "':'");
                TypePtr type = parseType();
                ArgRole role = ArgRole::Input;
                if (accept(Tok::LBrace)) {
                    Token roleTok = expect(Tok::Ident, "an argument role (input, param, const or state)");
                    if (!parseArgRole(roleTok.text, role)) {
                        throw errorAt(roleTok, "unknown argument role '" + roleTok.text + "'");
                    }
                    expect(Tok::RBrace, "'}'");
                }
                args.push_back({id, type, role});
            } while (accept(Tok::Comma));
            expect(Tok::RParen, "')'");
        }

        expect(Tok::Arrow, "'->'");
        std::vector<TypePtr> resultTypes;
        if (accept(Tok::LParen)) {
            if (!accept(Tok::RParen)) {
                do {
                    resultTypes.push_back(parseType());
                } while (accept(Tok::Comma));
                expect(Tok::RParen, "')'");
            }
        } else {
            resultTypes.push_back(parseType());
        }

        std::vector<TypePtr> argTypes;
        for (const auto& arg : args) argTypes.push_back(arg.type);
        IRBuilder builder;
        FunctionPtr func = builder.createFunction(name.text, argTypes, resultTypes);
        SymbolTable symbols;
        for (size_t i = 0; i < args.size(); ++i) {
            const ValuePtr& value = func->getArguments()[i];
            if (!symbols.emplace(args[i].tok.text, value).second) {
                throw errorAt(args[i].tok, "redefinition of '%" + args[i].tok.text + "'");
            }
            value->setName(args[i].tok.text);
            func->setArgRole(i, args[i].role);
        }

        expect(Tok::LBrace, "'{'");
        while (!accept(Tok::RBrace)) {
            if (tok_.kind == Tok::Eof) throw errorAt(tok_, "expected '}' to close function '@" + name.text + "'");
            parseStatement(*func->getEntryBlock(), symbols);
        }
        return func;
    }

    void parseStatement(Block& block, SymbolTable& symbols) {
        if (tok_.kind == Tok::Ident && tok_.text == "return") {
            consume();
            std::vector<ValuePtr> values;
            if (accept(Tok::LParen)) {
                if (!accept(Tok::RParen)) {
                    values = parseValueList(symbols, "a returned value");
                    expect(Tok::RParen, "')'");
                }
            } else if (tok_.kind == Tok::ValueId) {
                values = parseValueList(symbols, "a returned value");
            }
            block.addOperation(std::make_shared<Operation>("return", values, std::vector<TypePtr>{}));
            return;
        }

        Token result = expect(Tok::ValueId, "'%name = op(...)' or 'return'");
        expect(Tok::Equal, "'='");
        Token opTok = expect(Tok::Ident, "an op name");
        const OpDef* def = OpRegistry::instance().lookup(opTok.text);
        if (!def || !def->hasResult) throw errorAt(opTok, "unknown op '" + opTok.text + "'");

        expect(Tok::LParen, "'('");
        std::vector<ValuePtr> operands;
        if (!accept(Tok::RParen)) {
            operands = parseValueList(symbols, "an operand");
            expect(Tok::RParen, "')'");
        }
        if (def->numOperands >= 0 && operands.size() != static_cast<size_t>(def->numOperands)) {
            throw errorAt(opTok, "'" + opTok.text + "' expects " + std::to_string(def->numOperands) +
                                     " operand(s), got " + std::to_string(operands.size()));
        }

        TypePtr type = nullptr;
        if (accept(Tok::Colon)) type = parseType();
        auto op = std::make_shared<Operation>(opTok.text, operands, std::vector<TypePtr>{type});

        if (tok_.kind == Tok::LBrace) {
            for (const auto& [key, attr] : parseAttrDict()) {
                if (const AttrSpec* spec = def->findAttr(key)) {
                    op->setAttribute(key, coerce(attr, *spec, opTok.text));
                } else if (key.find('.') != std::string::npos) {
                    op->setAttribute(key, attr.value); // annotations such as fusion.group
                } else {
                    throw errorAt(attr.tok, "unknown attribute '" + key + "' for '" + opTok.text + "'");
                }
            }
        }

        try {
            finalizeOp(*op, /*inferResultType=*/type == nullptr);
        } catch (const IRError& e) {
            throw errorAt(opTok, e.what());
        }
        if (!symbols.emplace(result.text, op->getResult(0)).second) {
            throw errorAt(result, "redefinition of '%" + result.text + "'");
        }
        op->getResult(0)->setName(result.text);
        block.addOperation(op);
    }

    Lexer lexer_;
    Token tok_;
};

} // namespace

ModulePtr IRParser::parseModule(std::string_view source) {
    return Parser(source).parseModule();
}

FunctionPtr IRParser::parseFunction(std::string_view source) {
    ModulePtr module = parseModule(source);
    if (module->getFunctions().empty()) throw IRError("input contains no function");
    return module->getFunctions()[0];
}

} // namespace dlc
