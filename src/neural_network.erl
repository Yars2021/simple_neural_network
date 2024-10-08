-module(neural_network).

-export([set_input/2,
         set_expected/2,
         set_weights/2,
         get_weights/1,
         get_alpha/1,
         calculate/1,
         create_net/1,
         adjust_weights/1,
         run_test/3]).

-define(MAX_ALPHA, 1.0).
-define(MIN_ALPHA, 0.0).
-define(MAX_ERR, 0.1).


-record(neural_net, {
    input = [],
    layers = [],
    output = [],
    expected = [],
    delta = [],
    alpha = 0
}).


% Создать нейронную сеть по конфигурации
create_net([InputLayerConfig | LayersConfig]) ->
    #neural_net{
        input = [],
        layers = [create_input_layer(InputLayerConfig) | create_normal_layers([InputLayerConfig | LayersConfig])],
        output = [],
        expected = [],
        delta = [],
        alpha = 0
    }.


create_input_layer(0) -> [];
create_input_layer(N) -> [{0, []} | create_input_layer(N - 1)].


create_normal_layer(0, _) -> [];
create_normal_layer(N, P) ->
    [{0, [{Src, 2 * rand:uniform() - 1} || Src <- lists:seq(1, P)]} | create_normal_layer(N - 1, P)].


create_normal_layers([LastHiddenLayer | [OutputLayer | []]]) ->
    [create_normal_layer(OutputLayer, LastHiddenLayer)];

create_normal_layers([Prev | [Current | Tail]]) ->
    [create_normal_layer(Current, Prev) | create_normal_layers([Current | Tail])].


% Установка входных данных
set_input(#neural_net{} = Net, Input) ->
    Net#neural_net{input = lists:foldl(fun(Line, Acc) -> Acc ++ Line end, [], Input)}.


% Установка ожидаемых значений
set_expected(#neural_net{} = Net, Expected) ->
    Net#neural_net{expected = Expected}.


% Установить веса найронной сети
set_weights(#neural_net{} = Net, Weights) -> Net#neural_net{layers = Weights}.


% Получить данные о весах нейросети
get_weights(#neural_net{layers = Layers}) -> Layers.


% Перенос входного вектора на первый (входной) слой нейросети
transfer_inputs(#neural_net{input = Input, layers = [InputLayer | OtherLayers]} = Net) ->
    Net#neural_net{layers = [transfer_inputs(InputLayer, Input) | OtherLayers]}.


transfer_inputs([], []) -> [];
transfer_inputs([_ | InputLayerTail], [Input | InputTail]) ->
    [{Input, []} | transfer_inputs(InputLayerTail, InputTail)].


% Перенос последнего (выходного) слоя нейросети на выходной вектор
transfer_output(#neural_net{layers = Layers} = Net) ->
    Net#neural_net{layers = Layers, output = transfer_output(Layers, [])}.

transfer_output([[]], Acc) -> lists:reverse(Acc);
transfer_output([[{Output, _} | Tail]], Acc) ->
    transfer_output([Tail], [Output | Acc]);

transfer_output([_ | Tail], Acc) ->
    transfer_output(Tail, Acc).


% Функция активации
activation(X) -> 1 / (1 + math:exp(-X)).


% Подсчет ошибки
get_error(#neural_net{output = Output, expected = Expected}) ->
    get_error(Output, Expected, 0).

get_error([], [], Acc) -> Acc / 2;
get_error([Output | OutputTail], [Expected | ExpectedTail], Acc) ->
    get_error(OutputTail, ExpectedTail, Acc + abs(Output - Expected)).


% Пересчет alpha
get_alpha(#neural_net{output = Output} = Net) ->
    Net#neural_net{alpha = 2 * abs(get_error(Net)) / length(Output) * (?MAX_ALPHA - ?MIN_ALPHA) + ?MIN_ALPHA}.


% Вычисления для нейрона
calculate_neuron({_, Connections}, PrevLayer) -> {calculate_neuron(Connections, PrevLayer, 0), Connections}.

calculate_neuron([], _, Acc) -> activation(Acc);
calculate_neuron([{Src, Weight} | ConnectionsTail], PrevLayer, Acc) ->
    {Output, _} = lists:nth(Src, PrevLayer),
    calculate_neuron(ConnectionsTail, PrevLayer, Acc + Weight * Output).


% Вычисления для слоя
calculate_layer([], _) -> [];
calculate_layer([Neuron | CurrTail], PrevLayer) ->
    [calculate_neuron(Neuron, PrevLayer) | calculate_layer(CurrTail, PrevLayer)].


% Вычисление для нескольких слоев. Первый должен быть входным
calculate_layers([LastHiddenLayer | [OutputLayer | []]]) -> [calculate_layer(OutputLayer, LastHiddenLayer)];
calculate_layers([Prev | [Current | Tail]]) ->
    NewCurrent = calculate_layer(Current, Prev),
    [NewCurrent | calculate_layers([NewCurrent | Tail])].


% Вычисления для сети
calculate(Net) ->
    #neural_net{layers = [InputLayer | OtherLayers]} = LoadedInputNet = transfer_inputs(Net),
    LoadedInputNet#neural_net{layers = [InputLayer | calculate_layers([InputLayer | OtherLayers])]}.


% Пересчитать веса нейросети
adjust_weights(#neural_net{layers = Layers, output = Output, expected = Expected, delta = Delta, alpha = Alpha}) ->
    [_, OtherLayers] = lists:reverse(Layers),
    OutputDeltas = calc_output_deltas(Output, Expected).


calc_output_deltas([], []) -> [];
calc_output_deltas([Output | OutputTail]], [Expected | ExpectedTail]) ->
    [Output * (1 - Output) * (Expected - Output) | calc_output_deltas(OutputTail, ExpectedTail)].


calc_layer_deltas([], _) -> [];
calc_layer_deltas([{Output, Connections} | Tail], NextLayerDeltas) ->
    [Output * (1 - Output) * lists:foldl(fun({Src, Weight}, Acc) ->
        Acc + Weight * lists:nth(Src, NextLayerDeltas)
    end, 0, Connections) | calc_layer_delta(Tail, NextLayerDeltas)].


calc_network_deltas(ReversedLayers) -> calc_network_

calc_network_deltas([]) -> [];
calc_network_deltas([Layer | ReversedTail]) ->
    [calc_layer_deltas(Layer, ) | calc_network_deltas(ReversedTail)]


run_test(Config, Input, Expected) ->
    get_alpha(transfer_output(calculate(set_expected(set_input(create_net(Config), Input), Expected)))).
