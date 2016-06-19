
#include <time.h>
#include "sharedmatting.h"


using namespace cv;


#define  KI      10    //  D_IMAGE,  for expansion of known region
#define  KC      5.0   //  D_COLOR,  for expansion of known region
#define  KG      4     //  for sample gathering, each unknown p gathers at most kG forground and background samples
#define  EN      3
#define  EA      2
#define  EB      4


#define  IS_BACKGROUND(x)  (x == 0)
#define  IS_FOREGROUND(x)  (x == 255)
#define  IS_KNOWN(x)       (IS_BACKGROUND(x) || IS_FOREGROUND(x))
#define  IS_UNKNOWN(x)     (!IS_KNOWN(x))
#define  LOAD_RGB_SCALAR(data, pos)    Scalar(data[pos], data[pos+1], data[pos+2])


#pragma mark Public functions

SharedMatting::SharedMatting()
{
    unknownSet.clear();
    tuples.clear();
}

SharedMatting::~SharedMatting()
{
    pImg.release();
    matte.release();
    unknownSet.clear();
    tuples.clear();
    ftuples.clear();
    
    for (int i = 0; i < height; ++i)
    {
        delete[] m_ppTriData[i];
        delete[] unknownIndex[i];
        delete[] alpha[i];
    }
    delete[] m_ppTriData;
    delete[] unknownIndex;
    delete[] alpha;
}

void SharedMatting::loadImage(const char * filename)
{
    pImg = imread(filename);
    if (!pImg.data)
    {
        cout << "Loading Image Failed!" << endl;
        exit(-1);
    }
    height     = pImg.rows;
    width      = pImg.cols;
    step       = pImg.step1();
    channels   = pImg.channels();
    data       = (uchar *)pImg.data;
    unknownIndex  = new int*[height];
    m_ppTriData           = new int*[height];
    alpha         = new int*[height];
    for(int i = 0; i < height; ++i)
    {
        unknownIndex[i] = new int[width];
        m_ppTriData[i]          = new int[width];
        alpha[i]        = new int[width];
    }
    
    matte.create(Size(width, height), CV_8UC1);
}


//  Trimap value: 0 - background, 255 - foreground,  others(128) - unknown.
void SharedMatting::loadTrimap(const char * filename)
{
    cv::Mat trimap = imread(filename);
    if (!trimap.data) {
        cout << "Loading Trimap Failed!" << endl;
        exit(-1);
    }
    uchar * d   = (uchar *)trimap.data;
    
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            m_ppTriData[i][j] = d[i * step + j * channels];
        }
    }

    trimap.release();
}

void SharedMatting::save(const char * filename)
{
    imwrite(filename, matte);
}

void SharedMatting::solveAlpha()
{
    clock_t start, finish;
    clock_t begin, end;
    //expandKnown()
    begin = start = clock();
    cout << "Expanding...";
    expandKnown();
    cout << "    over!!!" << endl;
    finish = clock();
    cout <<  double(finish - start) / (CLOCKS_PER_SEC * 2.5) << endl;
    
    //gathering()
    start = clock();
    cout << "Gathering...";
    gathering();
    cout << "    over!!!" << endl;
    finish = clock();
    cout <<  double(finish - start) / (CLOCKS_PER_SEC * 2.5) << endl;
    
    //refineSample()
    start = clock();
    cout << "Refining...";
    refineSample();
    cout << "    over!!!" << endl;
    finish = clock();
    cout <<  double(finish - start) / (CLOCKS_PER_SEC * 2.5) << endl;
    
    //localSmooth()
    start = clock();
    cout << "LocalSmoothing...";
    localSmooth();
    cout << "    over!!!" << endl;
    end = finish = clock();
    cout <<  double(finish - start) / (CLOCKS_PER_SEC * 2.5) << endl;
    
    cout << "Total: " << double(end - begin)/(CLOCKS_PER_SEC*2.5) <<endl;
    
    getMatte();
}

#pragma mark Internal functions

void SharedMatting::expandKnown()
{
    vector<struct labelPoint> vp;
    int kc2 = KC * KC;

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            if ( IS_UNKNOWN(m_ppTriData[i][j]) ) {
                int label = -1;
                bool bLabel = false;
                Scalar p = LOAD_RGB_SCALAR(data, i*step+j*channels);
                
                for (int k = 1; (k <= KI) && !bLabel; ++k) {
                    int k1 = max(0, i - k);
                    int k2 = min(i + k, height - 1);
                    int l1 = max(0, j - k);
                    int l2 = min(j + k, width - 1);
                    
                    for (int l = k1; (l <= k2) && !bLabel; ++l) {
                        double dis;
                        double gray;
                        
                        gray = m_ppTriData[l][l1];
                        if (IS_KNOWN(gray)) {
                            dis = pixelDistance(Point(i, j), Point(l, l1));
                            if (dis > KI) {
                                continue;
                            }
                            Scalar q = LOAD_RGB_SCALAR(data, l*step + l1*channels);
                            
                            double distanceColor = colorDistance2(p, q);
                            if (distanceColor <= kc2) {
                                bLabel = true;
                                label = gray;
                            }
                        }
                        if (bLabel) {
                            break;
                        }
                        
                        gray = m_ppTriData[l][l2];
                        if (IS_KNOWN(gray)) {
                            dis = pixelDistance(Point(i, j), Point(l, l2));
                            if (dis > KI) {
                                continue;
                            }
                            Scalar q = LOAD_RGB_SCALAR(data, l*step + l2*channels);
                            
                            double distanceColor = colorDistance2(p, q);
                            if (distanceColor <= kc2) {
                                bLabel = true;
                                label = gray;
                            }
                        }
                    }
                    
                    for (int l = l1; (l <= l2) && !bLabel; ++l) {
                        double dis;
                        double gray;
                        
                        gray = m_ppTriData[k1][l];
                        if (IS_KNOWN(gray)) {
                            dis = pixelDistance(Point(i, j), Point(k1, l));
                            if (dis > KI) {
                                continue;
                            }
                            
                            Scalar q = LOAD_RGB_SCALAR(data, k1*step+l*channels);
                            
                            double distanceColor = colorDistance2(p, q);
                            if (distanceColor <= kc2) {
                                bLabel = true;
                                label = gray;
                            }
                        }
                        gray = m_ppTriData[k2][l];
                        if (IS_KNOWN(gray)) {
                            dis = pixelDistance(Point(i, j), Point(k2, l));
                            if (dis > KI) {
                                continue;
                            }
                            Scalar q = LOAD_RGB_SCALAR(data, k2*step+l*channels);
                            
                            double distanceColor = colorDistance2(p, q);
                            if (distanceColor <= kc2) {
                                bLabel = true;
                                label = gray;
                            }
                        }
                    }
                }
                if (label != -1) {
                    struct labelPoint lp;
                    lp.x = i;
                    lp.y = j;
                    lp.label = label;
                    vp.push_back(lp);
                } else {
                    Point lp;
                    lp.x = i;
                    lp.y = j;
                    unknownSet.push_back(lp);
                }
            }
        }
    }
    
    vector<struct labelPoint>::iterator it;
    for (it = vp.begin(); it != vp.end(); ++it)
    {
        int ti = it->x;
        int tj = it->y;
        int label = it->label;
        m_ppTriData[ti][tj] = label;
    }
}

double SharedMatting::comalpha(Scalar c, Scalar f, Scalar b)
{
    double alpha = ((c.val[0] - b.val[0]) * (f.val[0] - b.val[0]) +
                    (c.val[1] - b.val[1]) * (f.val[1] - b.val[1]) +
                    (c.val[2] - b.val[2]) * (f.val[2] - b.val[2]))
    / ((f.val[0] - b.val[0]) * (f.val[0] - b.val[0]) +
       (f.val[1] - b.val[1]) * (f.val[1] - b.val[1]) +
       (f.val[2] - b.val[2]) * (f.val[2] - b.val[2]) + 0.0000001);
    return min(1.0, max(0.0, alpha));
}

double SharedMatting::chromaticDistortion(int i, int j, Scalar f, Scalar b)
{
    Scalar c = LOAD_RGB_SCALAR(data, i*step + j*channels);
    
    double alpha = comalpha(c, f, b);
    
    double result = sqrt((c.val[0] - alpha * f.val[0] - (1 - alpha) * b.val[0]) * (c.val[0] - alpha * f.val[0] - (1 - alpha) * b.val[0]) +
                         (c.val[1] - alpha * f.val[1] - (1 - alpha) * b.val[1]) * (c.val[1] - alpha * f.val[1] - (1 - alpha) * b.val[1]) +
                         (c.val[2] - alpha * f.val[2] - (1 - alpha) * b.val[2]) * (c.val[2] - alpha * f.val[2] - (1 - alpha) * b.val[2]));
    return result / 255.0;
}

double SharedMatting::neighborhoodAffinity(int i, int j, Scalar f, Scalar b)
{
    int i1 = max(0, i - 1);
    int i2 = min(i + 1, height - 1);
    int j1 = max(0, j - 1);
    int j2 = min(j + 1, width - 1);
    
    double  result = 0;
    
    for (int k = i1; k <= i2; ++k)
    {
        for (int l = j1; l <= j2; ++l)
        {
            double distortion = chromaticDistortion(k, l, f, b);
            result += distortion * distortion;
        }
    }
    
    return result;
}

double SharedMatting::energyOfPath(int i1, int j1, int i2, int j2)
{
    double ci = i2 - i1;
    double cj = j2 - j1;
    double z  = sqrt(ci * ci + cj * cj);
    
    double ei = ci / (z + 0.0000001);
    double ej = cj / (z + 0.0000001);
    
    double stepinc = min(1 / (abs(ei) + 1e-10), 1 / (abs(ej) + 1e-10));
    double result = 0;
    
    Scalar pre = LOAD_RGB_SCALAR(data, i1*step + j1*channels);
    
    int ti = i1;
    int tj = j1;
    
    for (double t = 1; ;t += stepinc)
    {
        double inci = ei * t;
        double incj = ej * t;
        int i = int(i1 + inci + 0.5);
        int j = int(j1 + incj + 0.5);
        
        double z = 1;
        
        Scalar cur = LOAD_RGB_SCALAR(data, i*step + j*channels);
        
        if (ti - i > 0 && tj - j == 0)
        {
            z = ej;
        }
        else if(ti - i == 0 && tj - j > 0)
        {
            z = ei;
        }
        
        result += ((cur.val[0] - pre.val[0]) * (cur.val[0] - pre.val[0]) +
                   (cur.val[1] - pre.val[1]) * (cur.val[1] - pre.val[1]) +
                   (cur.val[2] - pre.val[2]) * (cur.val[2] - pre.val[2])) * z;
        pre = cur;
        
        ti = i;
        tj = j;
        
        if(abs(ci) >= abs(inci) || abs(cj) >= abs(incj))
            break;
        
    }
    
    return result;
}

double SharedMatting::probabilityOfForeground(Point p, vector<Point>& f, vector<Point>& b)
{
    double fmin = 1e10;
    vector<Point>::iterator it;
    for (it = f.begin(); it != f.end(); ++it) {
        double fp = energyOfPath(p.x, p.y, it->x, it->y);
        if (fp < fmin) {
            fmin = fp;
        }
    }
    
    double bmin = 1e10;
    for (it = b.begin(); it != b.end(); ++it) {
        double bp = energyOfPath(p.x, p.y, it->x, it->y);
        if (bp < bmin) {
            bmin = bp;
        }
    }
    return bmin / (fmin + bmin + 1e-10);
}

double SharedMatting::aP(int i, int j, double pf, Scalar f, Scalar b)
{
    Scalar c = LOAD_RGB_SCALAR(data, i*step + j*channels);
    double alpha = comalpha(c, f, b);
    
    return pf + (1 - 2 * pf) * alpha;
}

double SharedMatting::pixelDistance(Point s, Point d)
{
    return sqrt(double((s.x - d.x) * (s.x - d.x) + (s.y - d.y) * (s.y - d.y)));
}

double SharedMatting::gP(Point p, Point fp, Point bp, double distance, double probability)
{
    Scalar f = LOAD_RGB_SCALAR(data, fp.x*step + fp.y*channels);
    Scalar b = LOAD_RGB_SCALAR(data, bp.x*step + bp.y*channels);
    
    double tn = pow(neighborhoodAffinity(p.x, p.y, f, b), EN);
    double ta = pow(aP(p.x, p.y, probability, f, b), EA);
    double tf = distance;
    double tb = pow(pixelDistance(p, bp), EB);
    
    return tn * ta * tf * tb;
}

double SharedMatting::sigma2(Point p)
{
    int xi = p.x;
    int yj = p.y;
    int bc, gc, rc;
    bc = data[xi * step + yj * channels];
    gc = data[xi * step + yj * channels + 1];
    rc = data[xi * step + yj * channels + 2];
    Scalar pc = Scalar(bc, gc, rc);
    
    int i1 = max(0, xi - 2);
    int i2 = min(xi + 2, height - 1);
    int j1 = max(0, yj - 2);
    int j2 = min(yj + 2, width - 1);
    
    double result = 0;
    int    num    = 0;
    
    for (int i = i1; i <= i2; ++i)
    {
        for (int j = j1; j <= j2; ++j)
        {
            int bc, gc, rc;
            bc = data[i * step + j * channels];
            gc = data[i * step + j * channels + 1];
            rc = data[i * step + j * channels + 2];
            Scalar temp = Scalar(bc, gc, rc);
            result += colorDistance2(pc, temp);
            ++num;
        }
    }
    
    return result / (num + 1e-10);
    
}

double SharedMatting::colorDistance2(Scalar cs1, Scalar cs2)
{
    return (cs1.val[0] - cs2.val[0]) * (cs1.val[0] - cs2.val[0]) +
    (cs1.val[1] - cs2.val[1]) * (cs1.val[1] - cs2.val[1]) +
    (cs1.val[2] - cs2.val[2]) * (cs1.val[2] - cs2.val[2]);
}

void SharedMatting::sample(std::vector<vector<Point> > &foregroundSamples, std::vector<vector<Point> > &backgroundSamples)
{
    int   a,b,i;
    int   x,y,p,q;
    int   w,h,gray;
    int   angle;
    double z,ex,ey,t,step;
    vector<Point>::iterator iter;
    
    a=360/KG;
    b=1.7f*a/9;
    foregroundSamples.clear();
    backgroundSamples.clear();
    w=pImg.cols;
    h=pImg.rows;
    for(iter=unknownSet.begin();iter!=unknownSet.end();++iter) {
        vector<Point> fPts,bPts;
        
        x=iter->x;
        y=iter->y;
        angle=(x+y)*b % a;
        for(i=0;i<KG;++i) {
            bool f1(false),f2(false);
            
            z=(angle+i*a)/180.0f*3.1415926f;
            ex=sin(z);
            ey=cos(z);
            step=min(1.0f/(abs(ex)+1e-10f),
                     1.0f/(abs(ey)+1e-10f));
            
            for(t=0;;t+=step) {
                p=(int)(x+ex*t+0.5f);
                q=(int)(y+ey*t+0.5f);
                if(p<0 || p>=h || q<0 || q>=w)
                    break;
                
                gray=m_ppTriData[p][q];
                if(!f1 &&  IS_BACKGROUND(gray)) {
                    Point pt = Point(p, q);
                    bPts.push_back(pt);
                    f1=true;
                } else {
                    if(!f2 && IS_FOREGROUND(gray)) {
                        Point pt = Point(p, q);
                        fPts.push_back(pt);
                        f2=true;
                    } else {
                        if(f1 && f2) break;
                    }
                }
            }
        }
        
        foregroundSamples.push_back(fPts);
        backgroundSamples.push_back(bPts);
    }
}

void SharedMatting::gathering()
{
    vector<Point> f;
    vector<Point> b;
    vector<Point>::iterator it1;
    vector<Point>::iterator it2;
    
    vector<vector<Point> > foregroundSamples,backgroundSamples;
    
    
    sample(foregroundSamples, backgroundSamples);
    
    int index = 0;
    size_t size = unknownSet.size();
    
    for (int m = 0; m < size; ++m)
    {
        int i = unknownSet[m].x;
        int j = unknownSet[m].y;
        
        double probability = probabilityOfForeground(Point(i, j), foregroundSamples[m], backgroundSamples[m]);
        double gmin = 1.0e10;
        
        Point tf;
        Point tb;
        
        bool flag = false;
        
        for (it1 = foregroundSamples[m].begin(); it1 != foregroundSamples[m].end(); ++it1)
        {
            double distance = pixelDistance(Point(i, j), *(it1));
            for (it2 = backgroundSamples[m].begin(); it2 < backgroundSamples[m].end(); ++it2)
            {
                
                double gp = gP(Point(i, j), *(it1), *(it2), distance, probability);
                if (gp < gmin)
                {
                    gmin = gp;
                    tf   = *(it1);
                    tb   = *(it2);
                    flag = true;
                }
            }
        }
        
        struct Tuple st;
        st.flag = -1;
        if (flag)
        {
            st.flag   = 1;
            st.f      = LOAD_RGB_SCALAR(data, tf.x*step + tf.y*channels);
            st.b      = LOAD_RGB_SCALAR(data, tb.x*step + tb.y*channels);
            st.sigmaf = sigma2(tf);
            st.sigmab = sigma2(tb);
        }
        
        tuples.push_back(st);
        unknownIndex[i][j] = index;
        ++index;
    }
}

void SharedMatting::refineSample()
{
    ftuples.resize(width * height + 1);
    for (int i = 0; i < height; ++i)
    {
        for (int j = 0; j < width; ++j)
        {
            int b, g, r;
            b = data[i * step +  j* channels];
            g = data[i * step +  j * channels + 1];
            r = data[i * step +  j * channels + 2];
            Scalar c = Scalar(b, g, r);
            int indexf = i * width + j;
            int gray = m_ppTriData[i][j];
            if (gray == 0 )
            {
                ftuples[indexf].f = c;
                ftuples[indexf].b = c;
                ftuples[indexf].alphar = 0;
                ftuples[indexf].confidence = 1;
                alpha[i][j] = 0;
            }
            else if (gray == 255)
            {
                ftuples[indexf].f = c;
                ftuples[indexf].b = c;
                ftuples[indexf].alphar = 1;
                ftuples[indexf].confidence = 1;
                alpha[i][j] = 255;
            }
            
        }
    }
    vector<Point>::iterator it;
    for (it = unknownSet.begin(); it != unknownSet.end(); ++it)
    {
        int xi = it->x;
        int yj = it->y;
        int i1 = max(0, xi - 5);
        int i2 = min(xi + 5, height - 1);
        int j1 = max(0, yj - 5);
        int j2 = min(yj + 5, width - 1);
        
        double minvalue[3] = {1e10, 1e10, 1e10};
        Point * p = new Point[3];
        int num = 0;
        for (int k = i1; k <= i2; ++k)
        {
            for (int l = j1; l <= j2; ++l)
            {
                int temp = m_ppTriData[k][l];
                
                if (temp == 0 || temp == 255)
                {
                    continue;
                }
                
                int index = unknownIndex[k][l];
                Tuple t   = tuples[index];
                if (t.flag == -1)
                {
                    continue;
                }
                
                double m  = chromaticDistortion(xi, yj, t.f, t.b);
                
                if (m > minvalue[2])
                {
                    continue;
                }
                
                if (m < minvalue[0])
                {
                    minvalue[2] = minvalue[1];
                    p[2]   = p[1];
                    
                    minvalue[1] = minvalue[0];
                    p[1]   = p[0];
                    
                    minvalue[0] = m;
                    p[0].x = k;
                    p[0].y = l;
                    
                    ++num;
                    
                }
                else if (m < minvalue[1])
                {
                    minvalue[2] = minvalue[1];
                    p[2]   = p[1];
                    
                    minvalue[1] = m;
                    p[1].x = k;
                    p[1].y = l;
                    
                    ++num;
                }
                else if (m < minvalue[2])
                {
                    minvalue[2] = m;
                    p[2].x = k;
                    p[2].y = l;
                    
                    ++num;
                }
            }
        }
        
        num = min(num, 3);
        
        
        double fb = 0;
        double fg = 0;
        double fr = 0;
        double bb = 0;
        double bg = 0;
        double br = 0;
        double sf = 0;
        double sb = 0;
        
        for (int k = 0; k < num; ++k)
        {
            int i  = unknownIndex[p[k].x][p[k].y];
            fb += tuples[i].f.val[0];
            fg += tuples[i].f.val[1];
            fr += tuples[i].f.val[2];
            bb += tuples[i].b.val[0];
            bg += tuples[i].b.val[1];
            br += tuples[i].b.val[2];
            sf += tuples[i].sigmaf;
            sb += tuples[i].sigmab;
        }
        
        fb /= (num + 1e-10);
        fg /= (num + 1e-10);
        fr /= (num + 1e-10);
        bb /= (num + 1e-10);
        bg /= (num + 1e-10);
        br /= (num + 1e-10);
        sf /= (num + 1e-10);
        sb /= (num + 1e-10);
        
        Scalar fc = Scalar(fb, fg, fr);
        Scalar bc = Scalar(bb, bg, br);
        int b, g, r;
        b = data[xi * step +  yj* channels];
        g = data[xi * step +  yj * channels + 1];
        r = data[xi * step +  yj * channels + 2];
        Scalar pc = Scalar(b, g, r);
        double   df = colorDistance2(pc, fc);
        double   db = colorDistance2(pc, bc);
        Scalar tf = fc;
        Scalar tb = bc;
        
        int index = xi * width + yj;
        if (df < sf)
        {
            fc = pc;
        }
        if (db < sb)
        {
            bc = pc;
        }
        if (fc.val[0] == bc.val[0] && fc.val[1] == bc.val[1] && fc.val[2] == bc.val[2])
        {
            ftuples[index].confidence = 0.00000001;
        }
        else
        {
            ftuples[index].confidence = exp(-10 * chromaticDistortion(xi, yj, tf, tb));
        }
        
        
        ftuples[index].f = fc;
        ftuples[index].b = bc;
        
        
        ftuples[index].alphar = max(0.0, min(1.0,comalpha(pc, fc, bc)));
        //cvSet2D(matte, xi, yj, ScalarAll(ftuples[index].alphar * 255));
    }
    tuples.clear();
}

void SharedMatting::localSmooth()
{
    vector<Point>::iterator it;
    double sig2 = 100.0 / (9 * 3.1415926);
    double r = 3 * sqrt(sig2);
    for (it = unknownSet.begin(); it != unknownSet.end(); ++it)
    {
        int xi = it->x;
        int yj = it->y;
        
        int i1 = max(0, int(xi - r));
        int i2 = min(int(xi + r), height - 1);
        int j1 = max(0, int(yj - r));
        int j2 = min(int(yj + r), width - 1);
        
        int indexp = xi * width + yj;
        Ftuple ptuple = ftuples[indexp];
        
        Scalar wcfsumup = Scalar::all(0);
        Scalar wcbsumup = Scalar::all(0);
        double wcfsumdown = 0;
        double wcbsumdown = 0;
        double wfbsumup   = 0;
        double wfbsundown = 0;
        double wasumup    = 0;
        double wasumdown  = 0;
        
        for (int k = i1; k <= i2; ++k)
        {
            for (int l = j1; l <= j2; ++l)
            {
                int indexq = k * width + l;
                Ftuple qtuple = ftuples[indexq];
                
                double d = pixelDistance(Point(xi, yj), Point(k, l));
                
                if (d > r)
                {
                    continue;
                }
                
                double wc;
                if (d == 0)
                {
                    wc = exp(-(d * d) / sig2) * qtuple.confidence;
                }
                else
                {
                    wc = exp(-(d * d) / sig2) * qtuple.confidence * abs(qtuple.alphar - ptuple.alphar);
                }
                wcfsumdown += wc * qtuple.alphar;
                wcbsumdown += wc * (1 - qtuple.alphar);
                
                wcfsumup.val[0] += wc * qtuple.alphar * qtuple.f.val[0];
                wcfsumup.val[1] += wc * qtuple.alphar * qtuple.f.val[1];
                wcfsumup.val[2] += wc * qtuple.alphar * qtuple.f.val[2];
                
                wcbsumup.val[0] += wc * (1 - qtuple.alphar) * qtuple.b.val[0];
                wcbsumup.val[1] += wc * (1 - qtuple.alphar) * qtuple.b.val[1];
                wcbsumup.val[2] += wc * (1 - qtuple.alphar) * qtuple.b.val[2];
                
                
                double wfb = qtuple.confidence * qtuple.alphar * (1 - qtuple.alphar);
                wfbsundown += wfb;
                wfbsumup   += wfb * sqrt(colorDistance2(qtuple.f, qtuple.b));
                
                double delta = 0;
                double wa;
                if (m_ppTriData[k][l] == 0 || m_ppTriData[k][l] == 255)
                {
                    delta = 1;
                }
                wa = qtuple.confidence * exp(-(d * d) / sig2) + delta;
                wasumdown += wa;
                wasumup   += wa * qtuple.alphar;
            }
        }
        
        int b, g, r;
        b = data[xi * step +  yj* channels];
        g = data[xi * step +  yj * channels + 1];
        r = data[xi * step +  yj * channels + 2];
        Scalar cp = Scalar(b, g, r);
        Scalar fp;
        Scalar bp;
        
        double dfb;
        double conp;
        double alp;
        
        bp.val[0] = min(255.0, max(0.0,wcbsumup.val[0] / (wcbsumdown + 1e-200)));
        bp.val[1] = min(255.0, max(0.0,wcbsumup.val[1] / (wcbsumdown + 1e-200)));
        bp.val[2] = min(255.0, max(0.0,wcbsumup.val[2] / (wcbsumdown + 1e-200)));
        
        fp.val[0] = min(255.0, max(0.0,wcfsumup.val[0] / (wcfsumdown + 1e-200)));
        fp.val[1] = min(255.0, max(0.0,wcfsumup.val[1] / (wcfsumdown + 1e-200)));
        fp.val[2] = min(255.0, max(0.0,wcfsumup.val[2] / (wcfsumdown + 1e-200)));
        
        //double tempalpha = comalpha(cp, fp, bp);
        dfb  = wfbsumup / (wfbsundown + 1e-200);
        
        conp = min(1.0, sqrt(colorDistance2(fp, bp)) / dfb) * exp(-10 * chromaticDistortion(xi, yj, fp, bp));
        alp  = wasumup / (wasumdown + 1e-200);
        
        double alpha_t = conp * comalpha(cp, fp, bp) + (1 - conp) * max(0.0, min(alp, 1.0));
        
        alpha[xi][yj] = alpha_t * 255;
    }
    ftuples.clear();
}


void SharedMatting::getMatte()
{
    int h     = matte.rows;
    int w     = matte.cols;
    size_t s     = matte.step1();
    uchar* d  = (uchar *)matte.data;
    for (int i = 0; i < h; ++i)
    {
        for (int j = 0; j < w; ++j)
        {
            d[i*s+j] = alpha[i][j];
            
        }
    }
}