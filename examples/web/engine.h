#pragma once

#include <string>
#include <vector>
#include <tuple>

typedef std::vector<uint32_t> Tokens;

struct mg_connection;
struct mg_http_message;

class Request
{
protected:
    struct mg_connection *conn;
    std::string model;
    std::string user;

    Request(struct mg_connection *c)
        : conn(c)
    {
    }

    virtual void Prepare(mg_http_message * message);
    virtual void Process() = 0;

    Tokens Tokenize(const std::string & str);

    void Enqueue();

    virtual ~Request();
};

class CompletionRequestBase : public Request
{
protected:
    double temperature;
    double top_p;
    int n;
    bool stream;
    std::string stop;
    int max_tokens;
    double presence_penalty;
    double frequency_penalty;
    Tokens input;

public:
    void Prepare(mg_http_message * message) override;

    void Process() override;

    CompletionRequestBase(struct mg_connection *c)
        : Request(c)
    {
    }

    virtual ~CompletionRequestBase() {}
};

class CompletionRequest : public CompletionRequestBase
{
public:
    void Prepare(mg_http_message * message) override;
    
    void Process() override;

    CompletionRequest(struct mg_connection *c)
        : CompletionRequestBase(c)
    {
    }

    virtual ~CompletionRequest() {}
};

class ChatCompletionRequest : public CompletionRequestBase
{
protected:
    std::vector<std::tuple<Tokens, Tokens>> messages;

public:
    ChatCompletionRequest(struct mg_connection *c)
        : CompletionRequestBase(c)
    {
    }

    virtual ~ChatCompletionRequest() {}
};

class EmbeddingsRequest : public Request
{
protected:
    Tokens input;

public:
    EmbeddingsRequest(struct mg_connection *c)
        : Request(c)
    {
    }

    void Prepare(mg_http_message * message) override;

    void Process() override;

    virtual ~EmbeddingsRequest() {}
};

int engine_init(int argc, char **argv);
int engine_add_request(Request *req);