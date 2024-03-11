function X = triangulate(P1, P2, x1t, x2t)
    zerocol=zeros(3,1);
    X = [];
    %disp(x1t)
    for i=1:length(x1t)
        x1=x1t(i,:);
        x2=x2t(i,:);
        M=[P1 -x1' zerocol;P2 zerocol -x2'];
        [Ut,St,Vt]=svd(M);
        v=Vt(:,end);
        X=[X v(1:4,1)];
    end
end
