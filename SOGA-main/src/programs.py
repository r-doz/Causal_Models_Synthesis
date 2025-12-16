def get_program(name):
    if name == "bernoulli":
        x = """
        theta = uniform([0,1], 2);
        
        if theta < _p {
            y = 1.;
        } else { 
            y = 0.; 
        } end if;
        """

    elif name == "burglary":
        x = """
        theta = uniform([0,1], 2);
        if theta < _pe {
            earthquake = 1.;
        } else { 
            earthquake = 0.; 
        } end if;

        if theta < _pb {
            burglary = 1.;
        } else {
            burglary = 0.; 
        } end if;

        if earthquake == 0 {
            if burglary == 0 {
                alarm = 0;
            } else {
                alarm = 1;
            } end if;
        } else {
            alarm = 1;
        } end if;

        if earthquake == 1 {
            phoneWorking = gm([0.7, 0.3], [1., 0.], [0., 0.]);
        } else {
            phoneWorking = gm([0.99, 0.01], [1., 0.], [0., 0.]);
        } end if;

        if alarm == 1 {
            if earthquake == 1 {
                maryWakes = gm([0.8, 0.2], [1., 0.], [0., 0.]);
            } else {
                maryWakes = gm([0.6, 0.4], [1., 0.], [0., 0.]);
            } end if;
        } else {
            maryWakes = gm([0.2, 0.8], [1., 0.], [0., 0.]);
        } end if;

        if maryWakes == 1 {
            if phoneWorking == 1 {
                called = 1;
            } else {
                called = 0;
            } end if;
        } else {
            called = 0;
        } end if;"""

    elif name == "clickgraph":
        x = """
        simAll = uniform([0,1],2);

        if simAll < _p {
            sim = 1;
        } else {
            sim = 0;
        } end if;

        beta1 = uniform([0,1],2);
        if sim == 1 {
            beta2 = beta1;
        } else {
            beta2 = uniform([0,1],2);
        } end if;

        if  uniform([0,1],2) -  beta1 < 0 {
            click0 = 1;
        } else {
            click0 = 0;
        } end if;


        if  uniform([0,1],2) -  beta2 < 0 {
            click1 = 1;
        } else {
            click1 = 0;
        } end if;

        """

    elif name == "clinicaltrial":
        x = '''    
        probTreated = uniform([0,1],2);
        probContr = uniform([0,1],2);
        effect = uniform([0,1],2);
        pc = _pc;

        if effect < _pe {
            pc = _pt;
        } else {
            skip;
        } end if;

        if probContr - pc < 0 {
            ycontr = 1;
        } else {
            ycontr = 0;
        } end if;

        if probTreated < _pt {
            ytreated = 1;
        } else {
            ytreated = 0;
        } end if;
        '''

    elif name == "coinbias":
        x = '''
        bias = beta([_p1,_p2],2); 
        if uniform([0,1],2) - bias < 0 {
            y = 1;
        } else {
            y = 0;
        } end if;
        '''

    elif name == "grass":
        x = """
        if uniform([0,1],2) <  _pcloudy {
            rain = gm([0.8, 0.2], [1.,0.], [0.,0.]);
            sprinkler = gm([0.1, 0.9], [1.,0.], [0.,0.]);
        } else {
            rain = gm([0.2, 0.8], [1.,0.], [0.,0.]);
            sprinkler = gm([0.5, 0.5], [1.,0.], [0.,0.]);
        } end if;

        if uniform([0,1],2) < _p1 {
            if rain == 1 {
                wetRoof = 1;
            } else {
                wetRoof = 0;
            } end if;
        } else {
            wetRoof = 0;
        } end if;

        if uniform([0,1],2) < _p2 {
            if rain == 1 {
                or1 = 1;
            } else {
                or1 = 0;
            } end if;
        } else {
            or1 = 0;
        } end if;

        if uniform([0,1],2) < _p3 {
            if sprinkler == 1 {
                or2 = 1;
            } else {
                or2 = 0;
            } end if;
        } else {
            or2 = 0;
        } end if;

        if or1 == 0 {
            skip;
            if or2 == 0 {
                wetGrass = 0;
            } else {
                wetGrass = 1;
            } end if;
        } else {
            wetGrass = 1;
        } end if;

        """

    elif name == "murdermistery":
        x = """
        if uniform([0,1],2) < _palice {
            withGun = gm([0.03,0.97], [1.,0.], [0.,0.]);
        } else {
            withGun = gm([0.8,0.2], [1.,0.], [0.,0.]);
        } end if;"""

    elif name == "noisior":
        x = """
        if uniform([0,1],2) < _p0 {
            n1 = _p1;
            n21 = _p1;
        } else {
            n1 = _p2;
            n21 = _p2;
        } end if;

        if uniform([0,1],2) < _p4 {
            n22 = _p1;
            n33 = _p1;
        } else {
            n22 = _p2;
            n33 = _p2;
        } end if;

        if uniform([0,1],2) - n21 > 0 {
            if uniform([0,1],2) - n22 > 0 {
                n2 = 0;
            } else {
                n2 = 1;
            } end if;
        } else {
            n2 = 1;
        } end if;

        if uniform([0,1],2) - n1 < 0 {
            n31 = _p1;
        } else {
            n31 = _p2;
        } end if;

        if n2 == 1 {
            n32 = _p1;
        } else {
            n32 = _p2;
        } end if;

        if uniform([0,1],2) - n31 < 0 {
            if uniform([0,1],2) - n32 > 0 {
                if uniform([0,1],2) - n33 > 0 {
                    n3 = 0;
                } else {
                    n3 = 1;
                } end if;
            } else {
                n3 = 1;
            } end if;
        } else {
            n3 = 1;
        } end if;
        """

    elif name == "surveyunbiased":
        x = '''
        if uniform([0,1],2) < _bias1 {
            ansb1 = 1;
        } else {
            ansb1 = 0;
        } end if;
            
        if uniform([0,1],2) < _bias2 {
            ansb2 = 1;
        } else {
            ansb2 = 0;
        } end if;
        '''
    elif name == 'trueskills':
        x = '''
        skillA = gm([1],[_pa],[10]);
        skillB = gm([1],[_pb],[10]);
        skillC = gm([1],[_pc],[10]);

        perfA = gm([1],[0],[15])+skillA;
        perfB = gm([1],[0],[15])+skillB;
        perfC = gm([1],[0],[15])+skillC;
        '''    
        #observe(perfA-perfB > 0);
        #observe(perfA-perfC > 0);
        
    elif name == 'twocoins':
        x = '''
        if uniform([0,1],2) < _first {
            if uniform([0,1],2) < _second {
                both = 1;
            } else {
                both = 0;
            } end if;
        } else {
            both = 0;
        } end if;'''
    elif name == "altermu":
        x = '''
        w1 = gm([1.], [_p1], [5.]);
        w2 = gm([1.], [_p2], [5.]);
        w3 = gm([1.], [_p3], [5.]);

        mean = w1*w2;
        mean = 3*mean - w3;

        y = gm([1.], [0.], [1.]) + mean;
        '''

    elif name == "altermu2":
        x = '''
        w1 = uniform([-10, 10], 2);
        w2 = uniform([-10, 10], 2);
        y = w1 + w2 + gm([1.], [_muy], [_vary]);
        '''

    elif name == "normalmixtures":
        x = '''
        mu1 = gm([1.], [_p1], [1]);
        mu2 = gm([1.], [_p2], [1]);

        if uniform([0,1], 2) < _theta { 
            y = gm([1.],[0.],[1.]) + mu1; 
        } else { 
            y = gm([1.],[0.],[1.]) + mu2; 
        } end if; 
        '''

    elif name == "test":
        x = '''
        v = gm([1.], [_p1], [1]);
        if v > 0 {
            y = gm([1.], [_p2], [1]);
        } else {
            y = gm([1.], [-2.], [1]);
        } end if;
        '''
    
    elif name == "test2":
        x = '''
        v = gm([1.], [0.], [_sigma1]);
        if v < _p1 {
            y = 0;
        } else {
            y = 1.;
        } end if;
        '''

    elif name == "pid":
        x = '''
        array[51] ang;

        v = 0;            
        currAng = 0.5;   
        id = 0;        
        oldv = 0;     

        for i in range(51) {

            ang[i] = currAng;                     

            d = 3.14 - currAng;                    
            torq = _s0*d + _s1*v + _s2*id;          
            id = 0.9*id + 0.1*d;       
            oldv = v;              
            
            v = v + 0.01*torq + gauss(0, 0.25);
            currAng = currAng + 0.05*v + 0.05*oldv + gauss(0., 0.25);

            /*if currAng > 6.28 {
                currAng = currAng - 6.28;
            } else {
                if currAng < 0 {
                    currAng = currAng + 6.28;
                } else {
                    skip;
                } end if;
            } end if; */

        } end for;

        ang[50] = currAng;         '''
    
    else:
        raise ValueError("Program not recognized")

    return x