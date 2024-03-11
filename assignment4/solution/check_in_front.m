function [P2, X] = check_in_front(P1, P2s, Xs)
    largest = 0;
    P2 = [];
    X = [];
    for i = 1:length(P2s)
        x1 = P1 * Xs{i}';
        x2 = P2s{i} * Xs{i}';
        inFront = 0;
        for j = 1:length(x1(3,:))
            if x1(3,j) > 0 && x2(3,j) > 0
                inFront = inFront + 1;
            end
        end
        if inFront > largest
            largest = inFront;
            P2 = P2s{i};
            X = Xs{i}';
        end
    end
end