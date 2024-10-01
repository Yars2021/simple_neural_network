-module(neural_network).

-export([activation/1,
         set_input/2,
         set_expected/2,
         get_error/1,
         get_alpha/1,
         clear_net/1,
         run_test/0]).

-define(MAX_ALPHA, 1.0).
-define(MIN_ALPHA, 0.0).
-define(MAX_ERR, 0.1).


-record(neural_net, {
    input_layer = [],
    output_layer = [],
    hidden_layers = [],
    weights = [],
    expected = [],
    aplha = 0,
    delta = 0
}).


% Функция активации
activation(X) -> 1 / (1 + math:exp(-X)).


% Установка входных данных
set_input(#neural_net{} = Net, Input) ->
    Net#neural_net{input_layer = lists:foldl(fun(Line, Acc) -> Acc ++ Line end, [], Input)}.


% Установка ожидаемых значений
set_expected(#neural_net{} = Net, Expected) ->
    Net#neural_net{expected = Expected}


% Подсчет ошибки
get_error(#neural_net{output_layer = OutputLayer, expected = Expected}) ->
    get_error(OutputLayer, Expected, 0).

get_error([], [], Acc) -> Acc / 2;
get_error([Output | OutputTail], [Expected | ExpectedTail], Acc) ->
    get_error(OutputTail, ExpectedTail, Acc + math:abs(Output - Expected)).


% Пересчет alpha
get_alpha(#neural_net{output_layer = OutputLayer, expected = Expected}) ->
    Net#neural_net{alpha = 2 * math:abs(get_error(Net)) / length(OutputLayer) * (?MAX_ALPHA - ?MIN_ALPHA) + ?MIN_ALPHA}.


% Очистка значений
clear_net(#neural_net{
    input_layer = InputLayer, 
    output_layer = OutputLayer,
    hidden_layers = HiddenLayers,
    weights = Weights,
    expected = Expected,
    alpha = Alpha,
    delta = Delta
}) ->
    Net#neural_net{
        input_layer = lists:map(fun(_) -> 0 end, InputLayer),
        output_layer = lists:map(fun(_) -> 0 end, OutputLayer),
        hidden_layers = lists:map(fun(Layer) -> lists:map(fun({_, _}) -> {0, 0} end, Layer) end, HiddenLayers),
        weights = lists:map(fun({Src, Dst}) -> {Src, lists:map(fun({Node, _}) -> {Node, 0} end, Dst)} end, Weights),
        alpha = 0,
        delta = 0
    }.


run_test() ->
    Net = #neural_net{input_layer = []},
    set_input(Net, [[0, 1, 0], [0, 1, 0], [1, 1, 1]]).
