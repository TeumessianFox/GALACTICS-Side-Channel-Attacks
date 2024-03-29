#include <boost/program_options.hpp>
#include <Eigen/Dense>
#include "rlutil.h"

#include <iostream>
#include <iomanip>
#include <fstream>
#include "Setup.h"
#include "KeyGen.h"
#include "Sampler.h"
#include "Sign.h"
#include "Verify.h"
#include "Entropy.h"
#include <ctime>
#include <mpfr.h>
#include <fstream>
#include <vector>
#include <cmath>

using namespace Eigen;
namespace po = boost::program_options;

typedef Matrix<long, Dynamic, Dynamic> MatrixXl;
typedef Matrix<long, Dynamic, 1>       VectorXl;

long gradasc(VectorXd& v, const MatrixXd& m, const double norm) {
    ArrayXd  a    = ArrayXd::Ones(m.rows());
    VectorXd grad(v.size()), newv(v.size()), dv(v.size());
    
    const double c = 2./(sigma * sigma), phimin = 0.005, nu = 0.8;
    double phi, phi0 = 0.25;
    long iter = 0;

    double l, newl = -INFINITY;

    v = norm * (a.matrix().transpose() * m).transpose().normalized();
    a = (m * v).array();
    a = exp(-c * a);
    l = -log(1 + a).sum();

    for(iter = 0; iter < 20; iter++) {
	a = c / (1 + a.inverse());

	grad = a.matrix().transpose() * m;
	dv   = grad - v.dot(grad) * v / (norm*norm);
	dv   = norm * dv.normalized();

	for(phi = phi0; phi > phimin; phi *= nu) {
	    newv = cos(phi) * v + sin(phi) * dv;
	    a    = (m * newv).array();
	    a    = exp(-c * a);
	    newl = -log(1 + a).sum();

	    if(newl > l + 0.5 * phi * dv.dot(grad))
		break;
	}
	if(newl > l) {
	    v = newv;
	    l = newl;
	}
	if(phi < phi0)
	    phi0 = phi / nu;

	if(phi <= phimin)
	    break;
    }

    return iter;
}

void startscreen(const VectorXl& s1) {
    rlutil::cls();
    rlutil::hidecursor();

    rlutil::locate(2,2);
    std::cout << "Generated BLISS-" << CLASS << " secret key.";
    
    rlutil::locate(2,4);
    std::cout << "s1 = [";
    int x = 8, i;
    for(i=0; i<N; i++) {
	std::cout << std::setw(2) << s1[i] << " ";
	x += 3;
	if(x + 1 >= rlutil::tcols()) {
	    std::cout << "\n       ";
	    x = 8;
	}
    }
    std::cout << "]\n\n Press any key to launch the attack...";
    rlutil::anykey();
}

void updatescreen(const VectorXl& s1, const VectorXl& v1,
	long sigs, long matches) {
    rlutil::cls();
    rlutil::hidecursor();

    int x = 8, cols = rlutil::tcols() - 1, i;

    rlutil::locate(2,2);
    std::cout << "Generated " << std::setw(6) << sigs << " signatures.";
    rlutil::locate(cols - 31, 2);
    std::cout << "Recovered coefficients: " << std::setw(3) << matches 
	      << "/" << N;

    rlutil::locate(2,4);
    std::cout << "s1*= [";
    x = 8;

    for(i=0; i<N; i++) {
	if(v1[i] == s1[i])
	    rlutil::setColor(rlutil::LIGHTBLUE);

	std::cout << std::setw(2) << v1[i] << " ";
	rlutil::resetColor();
	x += 3;
	if(x >= cols) {
	    std::cout << "\n       ";
	    x = 8;
	}
    }
    std::cout << "]\a\n\n ";
    
    if(matches < N) {
	double mitmcost;
	mitmcost = (lgamma(0.5 * N + 1.) - lgamma(0.5 * matches + 1.)
	    - lgamma(0.5 * (N-matches) + 1.) + log(N))/log(2.);
	std::cout << "Cost of recovering full secret (MITM): ";
	if(mitmcost > 60.)
	    rlutil::setColor(rlutil::LIGHTRED);
	else if(mitmcost > 40.)
	    rlutil::setColor(rlutil::LIGHTBLUE);
	else
	    rlutil::setColor(rlutil::LIGHTGREEN);

	std::cout << "2^" << std::fixed << std::setprecision(1) 
	          << mitmcost << std::endl;
	rlutil::resetColor();
    }
    else
    {
	rlutil::setColor(rlutil::LIGHTGREEN);
	std::cout << "Attack complete! ";
	rlutil::resetColor();
	std::cout << "Press any key...\n\n";
	rlutil::anykey();
	//rlutil::cls();
	rlutil::showcursor();
    }
}

  
int  main(int argc, char *argv[3]) {
	
  long i;

  long nbSign;
  bool compress = false;

  std::string outfile = "stdout";

  po::options_description desc("Allowed options");
  desc.add_options()
    ("help", "help message")
    ("compress", "generate compressed signatures")
    ("out,o", po::value<std::string>(), "output file")
    ("num,n", po::value<long>(&nbSign)->default_value(1000000), "number of signatures to generate")
    ;

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (vm.count("help")) {
    std::cout << desc << "\n";
    return EXIT_SUCCESS;
  }

  if (vm.count("compress"))
    compress = true;

  if (vm.count("out")) {
    outfile.assign(vm["out"].as<std::string>());
    if(std::freopen(outfile.c_str(), "w", stdout) == NULL) {
      std::cerr << "Could not open file '" << outfile << "'."
        << std::endl;
      return EXIT_FAILURE;
    }
  }

  Setup setup;
  Entropy random;
  Sampler sampler(sigma, alpha_rejection, &random);

  KeyGen key(setup, &random);
  Sign sign(setup, &sampler, &random);
  Verify verify;
  std::string message = "Hello World!"; 
    
	

  int sign_flip = -100;  
  long int inter =-1;
  int s1key=-100,s2key=-100;
  int attack_mode =1;  // 0 without noise 1 : noisy data
  long indc[kappa], z1[N], z2[N], w[2*N], s[2*N];
  int error;

  Map<VectorXl> vecw1(w,N), vecw2(w+N,N);
  
  FILE*s1_File;
  FILE*s2_File;
  s1_File = fopen("s1_file.txt", "r");  
  s2_File = fopen("s2_file.txt", "r");  
  
  
  if(attack_mode==1){ 
   for(i=0; i<N; i++) {
       error = fscanf(s1_File, "%d,", &s1key);
       if(error<-1){
	      printf("Error while reading s1");
	      break;
       }
       s[i]=s1key;
            
       error = fscanf(s2_File, "%d,", &s2key);
       if(error<-1) {
	       printf("Error while reading s2");
	       break;
       }
       s[i+N]=s2key;	
    } 
  } else  {
  
     for(i=0; i<N; i++) { 
    s[i]   = NTL::to_long(NTL::coeff(key.sk.s1, i));
    s[i+N] = NTL::to_long(NTL::coeff(key.sk.s2, i)); 
  }
   }
  
 

  Map<VectorXl> vecs1(s,N), vecs2(s+N,N), vecz1(z1,N);
  Map<VectorXl> vecs(s,2*N), vecw(w,2*N);

  VectorXd      vecs1d = vecs1.template cast<double>();
  VectorXd      vecs2d = vecs2.template cast<double>();
  VectorXd      vecsd  = vecs.template cast<double>();
  double        s1norm = vecs1d.norm();

  VectorXl	vecw1acc = VectorXl::Zero(N);
  VectorXl	vecw2acc = VectorXl::Zero(N);
  VectorXl      vecv1(N), vecv2(N), vecv(2*N);
  VectorXd      vecv1d(N), vecv2d(N), vecvd(2*N);

  MatrixXd	matw   = MatrixXd::Zero(nbSign, 2*N);

  long		partmatch = -1, fullmatch = -1; 
 
  startscreen(vecs1);
 
 
  int l;   
  FILE *c_File;
  FILE *z1_File;
  FILE *z2_File;
  FILE *b_File;	  
  
  c_File = fopen("c_file.txt", "r");
  z1_File = fopen("z1_file.txt", "r");
  z2_File = fopen("z2_file.txt", "r");
  b_File = fopen("b_file.txt", "r");  

  
  if(attack_mode == 0)  { std::cout << " perfect data attack" << std::endl;  }
  
   
    for(i=0; i<nbSign; i++) {

	 
	if(attack_mode == 0)  { 
	 
    sign.signMessage(key.pk, key.sk, message, sign_flip); 
	for(int k = 0; k < kappa; k++){
			indc[k] = sign.signOutput.indicesC[k];
        }
		  
     for(int j = 0; j < N; j++) {
    z1[j] = sign.signOutput.z1[j];
    z2[j] = compress ? (sign.signOutput.z2Carry[j]<<dropped_bits) :
	sign.signOutput.z2[j]; 
	}	
  
	}
    else
    { 

		if (c_File == NULL || z1_File == NULL || b_File == NULL|| z2_File == NULL)
		{
			printf("Can't open file for reading.\n");
		}  
		
		
		//read b from the file b_File  
		error = fscanf(b_File, "%ld,", &inter ); 
	sign_flip=inter;
	
	     //read z1 from the file z_File 
		for(l=0;l< N; l++){ 
        error = fscanf(z1_File, "%ld,", &inter);
	if(error<-1) break;
z1[l]=inter;
 
		}  
		  
		 
		//read z2 from the file z_File
		for(l=0;l< N; l++){
        error = fscanf(z2_File, "%ld,", &inter);
	if(error<-2) break;
      z2[l]=inter;	
		 }
		 
		//read c from c_File

		for (l=0;l<kappa;l++){ 
		error = fscanf(c_File, "%ld, ", &inter);
	        if (error<-1) break;
	       	indc[l]=inter;	
		}
		
	
    }
	 
 
	for(int j = 0; j < N; j++) {
	long z1c = 0, z2c = 0;
	for(int k = 0; k < kappa; k++) {
	    long m = j + indc[k];
	assert( 0 <= m && m < 2*N );
	    if( m < N ) {
	    z1c += z1[m]; 
	    z2c += z2[m];
	    }
	    else {
	    z1c -= z1[m-N];
	    z2c -= z2[m-N];
	    }
	}
	w[j]  = z1c;
	w[j+N]= z2c;
    }
 
    vecw1acc += sign_flip * vecw1;
    vecw2acc += sign_flip * vecw2;

    matw.row(i) = sign_flip * vecw.template cast<double>();

    if(i%5000 == 0) {
      long matchs1;
      gradasc(vecv1d,  matw.topLeftCorner(i+1, N), s1norm);
  
      for(int j = 0; j < N; j++) {
  	vecv1d[j] = std::round(vecv1d[j]);
      }
      vecv1 = vecv1d.template cast<long>();
  
      matchs1 = 0;
  
      for(int j = 0; j < N; j++) {
  	if(vecv1[j] == s[j]) matchs1++;
      }
       

      updatescreen(vecs1, vecv1, i, matchs1);
      if(matchs1 >= N-8 && partmatch < 0)
	  partmatch = i;
     
      if(matchs1 >= N) {
	  fullmatch = i;
	  break;
      }
    }
  }

  fclose(c_File);
  fclose(b_File);
  fclose(z1_File);
  fclose(z2_File);
  fclose(s1_File);
  fclose(s2_File);
  
  std::cout << " BLISS-" << CLASS << std::endl;
  std::cout << " Full match: " << fullmatch << std::endl;
  std::cout << " Part match: " << partmatch << std::endl;

  return i;
}

