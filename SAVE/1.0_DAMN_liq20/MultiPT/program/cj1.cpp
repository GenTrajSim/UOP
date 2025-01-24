#include <iostream>
#include <fstream>
#include <complex>
#include <cmath>
#include <sstream>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include<thread>
#include <unistd.h>

using namespace std;

double PPP_zhuan ( double box, double d_1, double d_2 ) {
	//
	double abs_d;
	
	if ( (abs(d_1-d_2)) > (box/2) ){
  	if (((d_2)-d_1)<0) {
    	abs_d = (box) - ((d_1)-(d_2));
    }else{
    	abs_d = -((box) - ((d_2)-(d_1)));
    }

  }else{
    abs_d = (d_2-d_1);
  }
	
	return abs_d;
}

//int Xu[39334];
string typeXu[200000];
double x[200000];
double y[200000];
double z[200000];

void proc_cj(int a, int b, int atom_N, double boxx, double boxy, double boxz){
	ofstream ofs;
	ofs.open("cj1.txt", ios::app);
	for (int ii = a; ii< b; ii++) {
		int jishu = 0 ;
		int neigh_i[10];
		if ( typeXu[ii] == "   OW1" ) {////
			for ( int jj = 0; jj < atom_N; jj++ ) {
				if ( (typeXu[jj] == "   OW1")&&(jj!=ii) ) {////
					double sx=0;
					double sy=0;
					double sz=0;
					if ( (abs(x[ii]-x[jj])) > (boxx/2) ){
						sx = boxx - abs(x[ii]-x[jj]);
          }else{
            sx = abs(x[ii]-x[jj]);
          }
          if ( (abs(y[ii]-y[jj])) > (boxy/2) ){
						sy = boxy - abs(y[ii]-y[jj]);
          }else{
            sy = abs(y[ii]-y[jj]);
          }
          if ( (abs(z[ii]-z[jj])) > (boxz/2) ){
						sz = boxz - abs(z[ii]-z[jj]);
          }else{
            sz = abs(z[ii]-z[jj]);
          }
					//
					if ( ((sx*sx)+(sy*sy)+(sz*sz)) <= (0.35*0.35) ) {
						neigh_i[jishu] = jj;
						jishu ++;
					}
				}
			}
		}
	  ofs<<ii<<" "<<jishu<<" ";
		for (int icj = 0; icj < jishu; icj ++) {
			ofs<<" "<<neigh_i[icj];
		}
		ofs<<endl;		
		//
	}
	ofs.close();
}

int main()
{
    cout << "Hello world!" << endl;
    cout.precision(16);
    ifstream OpenFile("panding.gro", ios::in);
    if (OpenFile.fail())
    {
        cout<<"Can not open target file"<<endl;
        return 0;
    }
    std::string lineStr;
    double boxx;
    double boxy;
    double boxz;
    int ii = 0;
    int b_N = 0 ;
    int N_OW1 = 0;
    if (OpenFile)
    {
        int i = 0;
        //int ii = 0;
        while (getline(OpenFile,lineStr))
        {
            i++;
           //cout<<i<< ":" <<lineStr<<endl;
           if ( i == 2 ) {
                //
                string bN_0;
                istringstream bs(lineStr);
                bs>>bN_0;
                b_N = atoi(bN_0.c_str());
                cout << b_N <<endl;
           }

           if (i == (3+b_N)) {
                string box_x,box_y,box_z;
                istringstream ip(lineStr);
                ip>>box_x>>box_y>>box_z;
                boxx = atof(box_x.c_str());
                boxy = atof(box_y.c_str());
                boxz = atof(box_z.c_str());
                cout<<boxx<<endl;
            }
            ///
            if ((i >= 3) && (i < (3+b_N)) )
            {
                ii ++;
                string a1,a2,a3,a4,a5,a6;
                //istringstream is(lineStr);
                //is>>a1>>a2>>a3>>a4>>a5;
                a1 = lineStr.substr(0,8);
                a2 = lineStr.substr(9,6);//type
                a3 = lineStr.substr(15,5);//XU
                a4 = lineStr.substr(21,8);
                a5 = lineStr.substr(29,8);
                a6 = lineStr.substr(37,8);
                //cout<<a1<<","<<a2<<","<<a3<<","<<a4<<","<<a5<<endl;
                int id = ii-1;
                typeXu[id] = a2.c_str();
                //cout << typeXu[id] <<endl;
                x[id] = (atof(a4.c_str()));
                y[id] = (atof(a5.c_str()));
                z[id] = (atof(a6.c_str()));
                //if (typeXu[id-1] == "   OW1") { ii ++; }////
            }
        }
    }
    OpenFile.close();

    int t ;
    t = ii - 1 ;
    // cout<<"####"<<t<<"####" << "," << typeXu[t] << "," << x[t] << "," << y[t] << "," << z[t] <<endl ;
    // cout<<boxx<<":"<<endl;
    // cout<<boxy<<":"<<endl;
    // cout<<boxz<<":"<<endl;
    double const PI = acos(double(-1));
    // cout<<PI<<endl;
    //
    ofstream OUT_print ;
    ii = ii -1;
    OUT_print.open("cj1.txt",ios::trunc) ;
    OUT_print<<"";
    OUT_print.close();
    
    //
    thread th1(proc_cj,int(0*(ii+1)/20),int(1*(ii+1)/2),ii+1,boxx,boxy,boxz);
    thread th2(proc_cj,int(1*(ii+1)/20),int(2*(ii+1)/2),ii+1,boxx,boxy,boxz);
    th1.join();
    th2.join();
    ofstream OUT_print_lmp ;
    OUT_print_lmp.open("panding.lammpstrj");
    OUT_print_lmp<<"ITEM: TIMESTEP"<<endl;;
    OUT_print_lmp<<"0"<<endl;
    OUT_print_lmp<<"ITEM: NUMBER OF ATOMS"<<endl;
    OUT_print_lmp<<ii+1<<endl;
    OUT_print_lmp<<"ITEM: BOX BOUNDS pp pp pp"<<endl;
    OUT_print_lmp<<"0 "<<boxx*10<<endl;
    OUT_print_lmp<<"0 "<<boxy*10<<endl;
    OUT_print_lmp<<"0 "<<boxz*10<<endl;
    OUT_print_lmp<<"ITEM: ATOMS id type xu yu zu ice17 ice1c ice1h ice2 ice3 ice4 ice5 ice6 ice7 ice20"<<endl;
    //OUT_print_lmp<<""<<endl;
    for (int lmpi = 0; lmpi < (ii+1); lmpi++) {
    	OUT_print_lmp<<lmpi+1<<" "<<typeXu[lmpi]<<" "<<x[lmpi]*10<<" "<<y[lmpi]*10<<" "<<z[lmpi]*10<<endl;
    }
    OUT_print_lmp.close();
    //
    return 0;
}
