/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   www.cs.sandia.gov/~sjplimp/lammps.html
   Steve Plimpton, sjplimp@sandia.gov, Sandia National Laboratories

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

// c++_driver = simple example of how an umbrella program
//              can invoke LAMMPS as a library on some subset of procs
// Syntax: c++_driver P in.lammps
//         P = # of procs to run LAMMPS on
//             must be <= # of procs the driver code itself runs on
//         in.lammps = LAMMPS input script
// See README for compilation instructions

#include "mpi.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "math.h"
#include "iostream"
#include "fstream"


#include "lammps.h"         // these are LAMMPS include files
#include "input.h"
#include "atom.h"
#include "force.h"
#include "pair.h"
#include "neigh_list.h"
#include "library.h"
#include "group.h"
#include "domain.h"

#include "xrand.h"

#define QUOTE_(x) #x
#define QUOTE(x) QUOTE_(x)

#include "lmppath.h"
#include QUOTE(LMPPATH/src/lammps.h)
#include QUOTE(LMPPATH/src/library.h)
#include QUOTE(LMPPATH/src/input.h)
#include QUOTE(LMPPATH/src/modify.h)
#include QUOTE(LMPPATH/src/fix.h)
#include QUOTE(LMPPATH/src/fix_external.h)

using namespace LAMMPS_NS;
using namespace std;

// the struct info is the lammps standard to pass informations to the callback function through fixexternal.
// the callback function is called everytimestep to add the active forces etc
struct Info {
  int me;
  LAMMPS *lmp;
};
FixExternal *fixexternal;

bool CONTROL = false;
int num_reproduction = 0;

// this function is called every timestep to add the active force/torque on the bacteria
// in addition, the function check if a bacterium need to grow. If so, it interrupts the run command and flag it
void force_callback(void *, bigint, int, int *, double **, double **);

// this function extract the value of a variable defined in lammps.
// this way, all variables can be defined in the "in.setup" file
double get_single_variable(LAMMPS* lmp, const char* pippo);

// this function exectues all commands of the file "in.setup" and read the value of some variables defined there using the previous function
void setup(LAMMPS* lmp, int me);

// each bacterium is a molecule. This function creates the template, we will use to insert the molecules
void create_template_molecule(LAMMPS* lmp, int me, double EE[3]);

// update for AT MOST nstep integratim timestpes
// the updating loops stops at nsteps, or if a flag a set via fix halt
// specifically, the loop halts every time a molecule splits in two
void update(LAMMPS* lmp, int me, int lammps, int nsteps);

// t_run, t_tumble, together with viscosity, velocity and torque, fix the MSD
// t_eps is the timescale for the production of eps
double t_run, t_tumble, t_eps, t_reproduction, t_adhesion;

// production of eps
void produce_eps(LAMMPS* lmp);

// growing dynamics
int timestep_grow; // after this many timesteps a molecule attemps to become bigger
int timestep_reproduction;
int total_simulation_steps;
int timesteps;
int timestep_splitting_check_interval;
int step_dumps;

bool USE_EPS;

int touch(LAMMPS* lmp, int m, double radius, int index_father);

// maximun number of bacterium we can handle
int const Nmax_bacteria = 10000;
// number of bacterium in the simulation box
int num_bacteria = 0;
// number of eps particles
int numEPS = 0;
// num of frozen bacteria
int num_frozen = 0;
// elapsed time
double total_time = 0;



// dimensionality
int dimension;
// num of bonds above which a bacterium is frozen
int num_bond_freeze;
int max_bonds12 = 3;
int max_bonds22 = 5;

// bond stiffness for the bacterium
double Kbond_bacterium;

// dump image
float zoom; // zoom factof for dump image

bool use_callback = true;


// freeze the molecules that have many bact-eps bonds
void freeze_molecules(LAMMPS* lmp);


// when a bacterium reproduces, it is found (e.g. Wong) that the
// daughter cell has an high probability of detaching from the substrate.
// here we neglect this and fix the probability of the daughter cell to be attached
// to the surface to be 1, but one could change this parameter
double p_sibiling_bounded = 1;

double Lbox,Frun,Torque,viscosity,Velocity;
double Length_Molecule,Diameter,dt_integration;
double cutoff_11, cutoff_12, cutoff_22;
double sigma_11, sigma_12, sigma_22;
double MassEPS;
int bond_type_12, bond_type_22;
double Temperature; // temperature of eps particles
int NBeads_Molecule;

void open_dump(LAMMPS* lmp, int seconds);
void close_dump(LAMMPS* lmp);
void delete_bacterium(LAMMPS* lmp, int m);
void run_and_tumble();
void fix_times(int m);
int create_bacterium_and_mol(LAMMPS* lmp,double Rstart[3], double Rend[3]);

int const MaxEpsHarms = 10;

// this class handle the growing of the molecules
class C_Grow{
public:
  LAMMPS* lmp;  
  char buffer[1000];
  double** x;
  double** v;
  double** forces;
  imageint *image;
  int* ind;
  double unwrap_ns[3], unwrap_ne[3], vel[3], length, equilibrium_length, angle, x1, y1;
  int is, ie;
  int num_to_split, new_m;
  int timestep;
  int tosplit[Nmax_bacteria]; // list of the bacteria that are to be splitted
  int bonded[100]; // id of bonded particle of type 2
  int nbonded;
  int idbonded[100][2];
  double R1[100][2]; // position of eps atoms linked to a molecule that splits in two
  char line[1000];
  // functions
  void grow_molecules(LAMMPS* lmp);
  void setup();
  void enlarge_molecule(int n);
  void split_molecules();
  void create_group_bonded(int m);
  void add_bonds(int b1, int b2);
  void relax_locally(int n, int m);
}Grow;

// this class keep track of the bacteria.
// lammps does not know whereas a bacterium is in the running or in the tumbling state,
class Bact{
  public:
  int exist; // -1 don't exist;
  int nstart;
  int nend;
  int state; // 0 run, 1 tumble; -1 = never used
  int molecule_type; // polydisperse system, not currenty used
  int bond_type; // bond length - to grow the molecules
  int next_grow_timestep; // time at which the molecule should change bond type to grow
  int mol_id;
  double X[3];
  double t_born; // initial = 0;
  double t_transition;// initial = 0
  double t_mobility;// initial = 0
  double t_eps_production; // initial = 0;
  double t_reproduction; // initial = 0;
  double t_adhesion; // initial = 0;
  double min_reproduction_time; // initial = 0;
  bool produce_eps;
  double Torque, Force;
  double dr[3];
  double norm;
  double length;
  double frozen_bond_length;
  bool frozen;
  int num_EPS;
  int EPSend[MaxEpsHarms];
  int EPSstart[MaxEpsHarms];
  //  private:
  void increment_bond_type(){ bond_type++;}
  void set_bond_type(int bt){ bond_type = bt;}
  void set_molecule_type(int mt){ molecule_type = mt;}
  void set_next_grow_timestep(int gt){ next_grow_timestep = gt;}
}Bacterium[Nmax_bacteria];

class TimeMeasurement{
  public:
  char buffer[100];
  clock_t begin[10], end;
  
  void set_begin(int ind){ begin[ind] = clock();}
  void set_end(){ end = clock();}
  double elapsed_secs(int ind){ return ( double(end-begin[ind])/CLOCKS_PER_SEC);}

  double time_measurement[10];
  void add_time(LAMMPS* lmp,int ind){ 
    set_end(); time_measurement[ind] += elapsed_secs(ind);
    sprintf(buffer,"variable Time%d equal %f",ind,time_measurement[ind]); 
    lmp->input->one(buffer);
  }
  void reset(LAMMPS *lmp){ 
    for(int q = 0; q < 10; q++){  
      time_measurement[q] = 0;
      sprintf(buffer,"variable Time%d equal %f",q,time_measurement[q]); 
      lmp->input->one(buffer);
    };
  }
}TimeMeas;

// the code works in serial - it could be possible to make it run in parallel, but we have not tried to
int main(int narg, char **arg){
  system("rm dumps/*");
  init_random(4,4);
  // setup MPI and various communicators
  // driver runs on all procs in MPI_COMM_WORLD
  // comm_lammps only has 1st P procs (could be all or any subset)
  MPI_Init(&narg,&arg);
  MPI_Comm comm = MPI_COMM_WORLD;
  int me,nprocs;
  MPI_Comm_rank(comm,&me);
  MPI_Comm_size(comm,&nprocs);

  int nprocs_lammps = nprocs;
  if (nprocs_lammps > nprocs) {
    if (me == 0)
      printf("ERROR: LAMMPS cannot use more procs than available\n");
    MPI_Abort(MPI_COMM_WORLD,1);
  }

  int lammps;
  if (me < nprocs_lammps) lammps = 1;
  else lammps = MPI_UNDEFINED;
  MPI_Comm comm_lammps;
  MPI_Comm_split(MPI_COMM_WORLD,lammps,0,&comm_lammps);

  LAMMPS *lmp;
  if (lammps == 1) lmp = new LAMMPS(0,NULL,comm_lammps);


  // the setup function read and exectues the file "in.setup"
  // variables specified in this lamps script are converted in variables of this c++ program
  // that is, all parameter values should be specified in the in.setup file, so that there is no need to act on this program 
  setup(lmp, me);
  // bacteria are molecules; here we create a template (a file named "template.bacteria"), so that we can then insert them with the create_atoms command 
  double EE[3];
  EE[0] = Length_Molecule-Diameter;
  EE[1] = 0;
  EE[2] = 0;
  create_template_molecule(lmp,me,EE);
  // this function calculate the length of the bacteria corresponding to the different bond used. The actual length of a bacteria is compared to these lengths when growing a bacteria

  //create a bacteriumin horizontal position, at the center of the box
  double Rs[3];
  double Re[3];
  Rs[0] = 0.5*Lbox-0.5*(Length_Molecule-Diameter); Rs[1] = 0.5*Lbox; Rs[2] = 1;
  Re[0] = 0.5*Lbox+0.5*(Length_Molecule-Diameter); Re[1] = 0.5*Lbox; Re[2] = 1;
  create_bacterium_and_mol(lmp, Rs, Re);

  char buffer[1000];
  
  // we fix the time at which the bacterium changes motility status
  // the duration of the running and of tumbling phases have an exponential distribution, we time constant specified in "in.setup"
  run_and_tumble(); // fix the time of the transitions

//  update(lmp,me,lammps,int(1200.0/dt_integration)); // we do up to 1000 steps; the run command halts if a molecule needs to split


  // computation of the mean square diffusion
  // remember to fix can_stop = false to suppress reproduction when studying the MSD
  // LINE 1080
/*
  sprintf(buffer,"group center id %d",6); lmp->input->one(buffer);
  for(int q = 0; q < 100; q++){
    sprintf(buffer,"compute MSD center msd"); lmp->input->one(buffer);
    sprintf(buffer,"reset_timestep 0"); lmp->input->one(buffer);
    sprintf(buffer,"variable TIME_MSD equal step*dt"); lmp->input->one(buffer);
    sprintf(buffer,"variable cmsd equal c_MSD[1]+c_MSD[2]");lmp->input->one(buffer);
    sprintf(buffer,"thermo_style  custom step atoms temp pe ke press v_cmsd"); lmp->input->one(buffer);
    sprintf(buffer,"fix fixprint all print 10 '${TIME_MSD} ${cmsd}' file MSD/MSD_%d.txt screen no",q); lmp->input->one(buffer); 
    update(lmp,me,lammps,int(200.0/dt_integration)); 
    sprintf(buffer,"unfix fixprint"); lmp->input->one(buffer); 
    sprintf(buffer,"uncompute MSD"); lmp->input->one(buffer); 
  }
  return 0;
*/

//  sprintf(buffer,"reset_timestep 0"); lmp->input->one(buffer);

  TimeMeas.reset(lmp);
  sprintf(buffer,"rm growth*.dat"); system(buffer);
  open_dump(lmp,step_dumps);

//  double pe_b;

//  ofstream outfile("pe.dat");

  while( (num_bacteria < 10000) && (timesteps < total_simulation_steps)){
//    freeze_molecules(lmp); 

    TimeMeas.set_begin(1); 
    update(lmp,me,lammps,1000); // we do up to 1000 steps; the run command halts if a molecule needs to split
    TimeMeas.add_time(lmp,1);


//    cout << "########################################################################## Before growth " << endl;
//    sprintf(buffer,"run 0"); lmp->input->one(buffer);
//    pe_b = get_single_variable(lmp,"PE");
//    outfile << pe_b << "	";
    TimeMeas.set_begin(5); 
    Grow.grow_molecules(lmp);
    TimeMeas.add_time(lmp,5);
//    cout << "########################################################################## After growth " << endl;
//    sprintf(buffer,"run 0"); lmp->input->one(buffer); 
//    pe_b = get_single_variable(lmp,"PE");
//    outfile << pe_b << "	";

//    if(pe_b > 100){ cout << "PIPPOLINO " << pe_b << endl; getchar();}

    if(USE_EPS){
//      cout << "################################################## Before EPS " << endl;
//      sprintf(buffer,"run 0"); lmp->input->one(buffer);
//      pe_b = get_single_variable(lmp,"PE");
//      outfile << pe_b << "	";
      TimeMeas.set_begin(4); produce_eps(lmp); TimeMeas.add_time(lmp,4);
//      cout << "################################################## After EPS " << endl;
//      sprintf(buffer,"run 0"); lmp->input->one(buffer); 
//      pe_b = get_single_variable(lmp,"PE");
//      outfile << pe_b;
//      getchar();
    }
//    outfile << endl;

//    if( (num_bacteria/50.0) == int(num_bacteria/50)){
//      lmp->input->file("in.dump_network");
//      sprintf(buffer,"mv Status/Pos.dat Status/Pos_N%d.dat",num_bacteria); system(buffer);
//      sprintf(buffer,"mv Status/Bonds.dat Status/Bonds_N%d.dat",num_bacteria); system(buffer);
//    }
  }
  close_dump(lmp);
 
//  outfile.close(); 

//  lmp->input->file("in.dump_network");
//  sprintf(buffer,"mv Status/Pos.dat Status/Pos_N%d.dat",num_bacteria); system(buffer);
//  sprintf(buffer,"mv Status/Bonds.dat Status/Bonds_N%d.dat",num_bacteria); system(buffer);
  // for record the trail count
  MPI_Finalize();
}


// dump every x seconds
void open_dump(LAMMPS* lmp, int every){
  char run[1024];
  sprintf(run,"dump WRITE all custom %d dumps/dumpfile.*.txt x y z type mol",every);
  lmp->input->one(run);
  sprintf(run,"dump_modify WRITE pad 10");
  lmp->input->one(run);
  sprintf(run,"dump_modify WRITE sort id");
  lmp->input->one(run);

//  sprintf(run,"dump mydump all image %d images/dump.*.jpg type type zoom %f size 1024 1024 bond atom type",every,zoom); lmp-> input->one(run);
//  sprintf(run,"dump_modify mydump backcolor white boxcolor black pad 10 adiam 1 ${cut11} adiam 2 ${cut22} bdiam 1 ${cut11} bdiam ${BondType12}*${BondType22} ${cut22}"); lmp-> input->one(run);

//   if(USE_EPS){
// show small particles to see the bonds
//    sprintf(run,"dump mydump_network all image %d network/dump.*.jpg type type zoom %f size 1024 1024 bond atom type center s 0.5 0.5 0.1",every,zoom*2); lmp-> input->one(run);
//    sprintf(run,"dump_modify mydump_network backcolor white boxcolor black pad 10 adiam 1 0.1 adiam 2 0.1 bdiam 1 0.2 bdiam ${BondType12}*${BondType22} 0.2"); lmp-> input->one(run);
//   }

  if(USE_EPS){
    sprintf(run,"fix fixprint all print %d \"${elapsed_time} ${num_bacteria} ${num_eps} ${NumFrozen} ${Time0} ${Time1} ${Time2} ${Time3} ${Time4} ${Time5}\" append growth_TR${t_reproduction}_EPS${t_eps}_TH${t_adhesion}.dat",every);
  }else{
    sprintf(run,"fix fixprint all print %d \"${elapsed_time} ${num_bacteria}\" append growth_TR${t_reproduction}_CUTOFF${CUTOFF_FACTOR}_EPS${eps11}_TH${t_adhesion}.dat",every);

  }
  lmp->input->one(run);
}

void close_dump(LAMMPS* lmp){
  char run[1024];
  sprintf(run,"undump WRITE"); lmp->input->one(run);
//  sprintf(run,"undump mydump"); lmp->input->one(run);
//  if(USE_EPS){
//    sprintf(run,"undump mydump_network"); lmp->input->one(run);
//  }
}



void update(LAMMPS* lmp, int me, int lammps, int nsteps){

  char run[1024];
  int ifix;


  sprintf(run,"group bact type 1"); lmp->input->one(run);
  sprintf(run,"group eps type 2"); lmp->input->one(run);
  // identify the eps atoms close to the surface

  sprintf(run,"thermo_style    custom step atoms pe ke press ebond");lmp->input->one(run);


  timesteps = get_single_variable(lmp,"TimeStep");
  if(numEPS > 1){
    sprintf(run,"velocity eps create %g %d dist gaussian",Temperature,timesteps*123+321+timesteps); lmp->input->one(run);
    sprintf(run,"velocity eps set NULL NULL 0.0"); lmp->input->one(run);
  }

//  if(USE_EPS){
//    sprintf(run,"fix BondEpsBac all bond/create ${timestep_bond} 1 2 ${cut12} ${BondType12} iparam %d 1 jparam %d 2 prob ${prob_bond12} %d",max_bonds12,max_bonds12,timesteps+23); lmp->input->one(run);
//    sprintf(run,"fix BondEpsEps eps bond/create ${timestep_bond} 2 2 ${cut22} ${BondType22} iparam %d 2 jparam %d 2 prob ${prob_bond12} %d",max_bonds22,max_bonds22, timesteps+123); lmp->input->one(run);
//    sprintf(run,"fix DeleteBondEpsBat all bond/break ${timestep_bond} 2 %f",1.5*cutoff_12); lmp->input->one(run);
//    sprintf(run,"fix DeleteBondEpsEps eps bond/break ${timestep_bond} 2 %f",1.5*cutoff_22); lmp->input->one(run);
//  }


  sprintf(run,"group bodies molecule != 0"); lmp->input->one(run);
  sprintf(run,"neigh_modify exclude molecule/intra bodies"); lmp->input->one(run); // if we exclude this, then no eps-eps bond are formed!
  sprintf(run,"fix Langevin eps langevin %g %g %g %d",Temperature,Temperature,100*dt_integration,timesteps*123+123); lmp->input->one(run);
  sprintf(run,"fix viscous bact viscous %g",viscosity); lmp->input->one(run);
  sprintf(run,"fix NVE all nve/limit ${max_displacement_one_step}"); lmp->input->one(run);
  if(dimension == 2){ sprintf(run,"fix 2D all enforce2d"); lmp->input->one(run);}


  // the callback function fixes StopRun=1 if a molecule needs to reproduce
  sprintf(run,"variable StopRun equal 0"); lmp->input->one(run);


  // fixexternal is a fix that calls at every integration timestep a function we define
  // in our case, this function is force_callback
  Info info;
  info.me = me;
  info.lmp = lmp;
  if(use_callback){
    sprintf(run,"fix CALLBACK bact external pf/callback 1 1"); lmp->input->one(run);
    // find the id of the fix CALLBACK
    ifix = lmp->modify->find_fix("CALLBACK");
    // find the pointer associated to this fix
    fixexternal = (FixExternal *) lmp->modify->fix[ifix];
    // clarify which is the function to be called at every timestep
    fixexternal->set_callback(force_callback,&info);
  }
  sprintf(run,"fix FixHalt all halt 1 v_StopRun > 0 error continue"); lmp->input->one(run);

  TimeMeas.set_begin(3); 
  sprintf(run,"run %d", nsteps); lmp->input->one(run);
  TimeMeas.add_time(lmp,3);

  sprintf(run,"unfix FixHalt"); lmp->input->one(run);
  sprintf(run,"unfix NVE"); lmp->input->one(run);
// eps
  sprintf(run,"unfix Langevin"); lmp->input->one(run);
  sprintf(run,"unfix viscous"); lmp->input->one(run);
//  if(USE_EPS){
//    sprintf(run,"unfix BondEpsBac"); lmp->input->one(run);
//    sprintf(run,"unfix BondEpsEps"); lmp->input->one(run);
//    sprintf(run,"unfix DeleteBondEpsBat"); lmp->input->one(run);
//    sprintf(run,"unfix DeleteBondEpsEps"); lmp->input->one(run);
//  }

  if(use_callback){ sprintf(run,"unfix CALLBACK"); lmp->input->one(run);}
  if(dimension == 2){ sprintf(run,"unfix 2D"); lmp->input->one(run);}
}

int index_of_new_bacterium(int molecule_type){
  int m = 0;
  bool keep_loop = true;
  bool never_used, correct_type;
  while(keep_loop){
    if( Bacterium[m].exist < 0 ){ // free bacterium
      if( Bacterium[m].molecule_type == -1) never_used = true; else never_used = false;
      if( Bacterium[m].molecule_type == molecule_type) correct_type = true; else correct_type = false;
      if( (never_used) || ( (never_used==false) && correct_type) ) keep_loop = false;
    }
    if(keep_loop) m++;
    if(m >= Nmax_bacteria){ cout << "Too many bacteria; increase Nmax_bacteria; I am about to chrash. Good bye;" << endl; getchar(); return 0; };
  }
  return m;
}

int create_bacterium_and_mol(LAMMPS* lmp,double Rstart[3], double Rend[3]){
  int m =  index_of_new_bacterium(1);
  int natoms = static_cast<int> (lmp->atom->natoms);
  double R[3];
  char line[100];
  Bacterium[m].nstart = natoms+1;
  Bacterium[m].nend = natoms+NBeads_Molecule;
  Bacterium[m].molecule_type = 1;
  Bacterium[m].exist = 1;
  Bacterium[m].frozen = false;
  Bacterium[m].num_EPS = 0;
  num_bacteria++;

  double a;
  for(int q = 0; q < 3; q++){
    a = Lbox*int(Rstart[q]/Lbox);
    Rstart[q] -= a;
    Rend[q] -= a;
  }

  // create the molecule
  sprintf(line,"create_atoms 0 single %lf %lf %lf mol TemplateBacteria 1", 0.5*Lbox, 0.5*Lbox, 0.0);  lmp->input->one(line);
  int index;
  double unwrap[3];
  double** x = lmp->atom->x; 
  imageint *image = lmp->atom->image;
  for(int q = 0; q < NBeads_Molecule; q++){
    index = lmp->atom->map(Bacterium[m].nstart+q); 
    lmp->domain->unmap(x[index],image[index],unwrap);
    for(int i = 0; i < 3; i++){
       R[i] = Rstart[i]+(Rend[i]-Rstart[i])*q*1.0/(NBeads_Molecule-1) - unwrap[i]; 
    }
    sprintf(line,"group atomo id %d", Bacterium[m].nstart+q); lmp->input->one(line);
    sprintf(line,"displace_atoms atomo move %lf %lf %lf", R[0], R[1], R[2]); lmp->input->one(line); 
    sprintf(line,"group atomo delete"); lmp->input->one(line);
  }
  return m; 
}

// this class takes care of the splitting
void C_Grow::setup(){
  x = lmp->atom->x;
  forces = lmp->atom->f;
  v = lmp->atom->v;
  image = lmp->atom->image;
  sprintf(line,"index");  
  int pp;
  int idx = lmp->atom->find_custom(line,pp);
  ind = lmp->atom->ivector[idx];
  timestep = get_single_variable(lmp,"TimeStep");
}

void C_Grow::grow_molecules(LAMMPS* lmp_instance){
  lmp = lmp_instance;
  setup();
  if(num_to_split > 0) split_molecules();
  num_to_split = 0;
  return;
}

void freeze_molecules(LAMMPS* lmp){
  int id, index, num_bond;
  double dx, dy, dz;
  double** x = lmp->atom->x;
  char line[1000];
  int max_num_bond = 0;
  for(int m = 0; m < num_bacteria; m++){
    if(Bacterium[m].frozen == false){
      num_bond = 0;
      for(int q = 0; q < NBeads_Molecule; q++){
        id = Bacterium[m].nstart+q; // global id of bonded atom;
        index = lmp->atom->map(id); // local id
        num_bond += lmp->atom->num_bond[index];
      }
      // subtract the number of bonds intra-molecule
      num_bond -= (NBeads_Molecule-1);  
      if(num_bond > num_bond_freeze){
        Bacterium[m].frozen = true;
        num_frozen++;
        sprintf(line,"variable NumFrozen equal %d",num_frozen); lmp->input->one(line);

        dx = x[ lmp->atom->map(Bacterium[m].nstart)][0]-x[ lmp->atom->map(Bacterium[m].nend)][0];
        dy = x[ lmp->atom->map(Bacterium[m].nstart)][1]-x[ lmp->atom->map(Bacterium[m].nend)][1];
        dz = x[ lmp->atom->map(Bacterium[m].nstart)][2]-x[ lmp->atom->map(Bacterium[m].nend)][2];
        Bacterium[m].frozen_bond_length = sqrt(dx*dx+dy*dy+dz*dz)/(NBeads_Molecule-1);
      }
      if(num_bond > max_num_bond) max_num_bond = num_bond;
    }
  }
  sprintf(line,"variable MaxNumMond equal %d",max_num_bond); lmp->input->one(line);
}

// Rs and Re are in unwrapped format
void move(LAMMPS* lmp, int m, double Rs[3], double Re[3]){
  char line[1000];
  int ns = Bacterium[m].nstart;
  int ne = Bacterium[m].nend;
  int n;
  double** x = lmp->atom->x;
  imageint *image = lmp->atom->image;
  double unwrap[3];
  double delta[3]; // displacement to put the molecule in the center of the box

  for(int i = ns; i <= ne; i++){
    n = lmp->atom->map(i);
    lmp->domain->unmap(x[n],image[n],unwrap);
    for(int q = 0; q < 3; q++){
      delta[q] = Rs[q]+(Re[q]-Rs[q])*(i-ns)*1.0/(ne-ns); // desired position
      delta[q] = delta[q]-unwrap[q];
    }
    sprintf(line,"group atom_move id %d", i); lmp->input->one(line);
    sprintf(line,"displace_atoms atom_move move %f %f 0.0",delta[0], delta[1]);lmp->input->one(line); // place the particle in the middle of the box
//    sprintf(line,"set atom %d x %f y %f z 0.0",i, delta[0], delta[1]);lmp->input->one(line); // place the particle in the middle of the box
    sprintf(line,"group atom_move delete"); lmp->input->one(line);
  }
}

void draw_molecule(LAMMPS* lmp, int m, int flag){
  ofstream outfile;
  if(flag == 0) outfile.open("father.dat");
  if(flag == 1) outfile.open("daughter1.dat");
  if(flag == 2) outfile.open("daughter2.dat");
  double** x = lmp->atom->x;
  imageint *image = lmp->atom->image;
  double unwrap[3];
  int index;
  for(int q = 0; q < NBeads_Molecule; q++){
    index = lmp->atom->map(Bacterium[m].nstart+q); // local id
    lmp->domain->unmap(x[index],image[index],unwrap);
    for(int q = 0; q <= 50; q++){
//       outfile << x[index][0]+0.5*cutoff_11*cos(q*2*M_PI/50) << "	" << x[index][1]+0.5*cutoff_11*sin(q*2*M_PI/50) << "	";
       outfile << unwrap[0]+0.5*cutoff_11*cos(q*2*M_PI/50) << "	" << unwrap[1]+0.5*cutoff_11*sin(q*2*M_PI/50) << endl;
    }
    outfile << endl << endl;
  }
  outfile.close(); 
}
      
bool touch(LAMMPS* lmp, int m, double Rs[3], double Re[3]){

  return false;

  if( (dimension == 3) && ( (Rs[2] < 0.5) || (Re[2] < 0.5)) ){
     return true;
  }
  int natoms = static_cast<int> (lmp->atom->natoms);
  int ns = Bacterium[m].nstart;
  int ne = Bacterium[m].nend;
  double** x = lmp->atom->x; 
  int index, type;
  double R[3], dist2;
  double cutsq[3];
  cutsq[1] = powl(cutoff_11,2);
  cutsq[2] = powl(cutoff_12,2);
  for(int n = 1; n <= natoms; n++){
    if( (n < ns) || (n > ne)){
      index = lmp->atom->map(n); // index of the atom we are considering
      type = lmp->atom->type[index]; // type of the atom we are considering
      for(int q = 0; q < NBeads_Molecule; q++){ // we loop over all molecules to see if there is a contact
        dist2 = 0;
        for(int k = 0; k < dimension; k++){ 
          R[k] = Rs[k] + q*1.0*(Re[k]-Rs[k])/(NBeads_Molecule-1); // the position of the beads varies between R[s] and R[e]
// we are using unwrapped coordintes for R, we need to move the coordinate in the box
          R[k] = R[k]-Lbox*int(R[k]/Lbox);
	  if(R[k] > 0) R[k] += Lbox; else if(R[k] < 0) R[k] += Lbox;
	  R[k] -= x[index][k]; // distance;
	  R[k] = fabs(R[k]); // absolute value
	  if(R[k] > 0.5*Lbox) R[k] = Lbox-R[k]; // periodic boundary conditions
          dist2 += R[k]*R[k]; // square distance
        }
        if( dist2 < cutsq[type] ){
//        if( dist2 < cutsq[type]*1.1 ){
//          cout << "Touching problem " << dist2 << endl; // dist2 << " " < cutsq[type] << endl;
//          cout << "Touching problem cutsq " << cutsq[type] << endl; // dist2 << " " < cutsq[type] << endl;
          return true;
        }
      }
    }
  }
  return false; 
}


void C_Grow::split_molecules(){
  int m;
  double Rs[3], Rc[3], Re[3], d1, d2, Rstart1[3], Rend1[3], Rstart2[3], Rend2[3], dist;
  bool write_separation = false; 
  bool is_touching;
  int inserted = 0;
  for(int q = 0; q < num_to_split; q++){
//    sprintf(buffer,"run 0"); lmp->input->one(buffer);
//    cout << "############################ ENERGY BEFORE SPLITTING " << get_single_variable(lmp,"PE") << endl;
    m = tosplit[q];
    // compute the angle of the bacterium wrt to x-axis
    is = lmp->atom->map(Bacterium[m].nstart); // local id
    ie = lmp->atom->map(Bacterium[m].nend); // local id
    lmp->domain->unmap(x[is],image[is],unwrap_ns);
    lmp->domain->unmap(x[ie],image[ie],unwrap_ne);
    d1 = 0; d2 = 0;
    for(int k = 0; k < dimension; k++){
      // position of the first and of the last bead
      Rs[k] = unwrap_ns[k]; 
      Re[k] = unwrap_ne[k];
      // central position;
      Rc[k] = 0.5*(Rs[k]+Re[k]);
      d1 += powl(Rs[k]-Rc[k],2);
      d2 += powl(Re[k]-Rc[k],2);
    }
    d1 = sqrt(d1);
    d2 = sqrt(d2);

    dist = sqrt( powl(Rs[0]-Re[0],2)+powl(Rs[1]-Re[1],2) );

    // if the final length of the two molecules is > initial length; it could not be the case if the molecule bends
    for(int q = 0; q < dimension; q++){
      Rstart1[q] = Rs[q];
      Rend1[q] = Rstart1[q] + (Length_Molecule-Diameter)*(Re[q]-Rs[q])/dist;

      Rend2[q] = Re[q];
      Rstart2[q] = Re[q]+(Length_Molecule-Diameter)*(Rs[q]-Re[q])/dist;

    }

    d1 = d2 = 0;
    for(int q = 0; q < dimension; q++){
      d1 += powl(Rstart1[q]-Rend1[q],2);
      d2 += powl(Rstart2[q]-Rend2[q],2);
    }
    d1 = sqrt(d1);
    d2 = sqrt(d2);
    // if the final length of the two molecules is > initial length; it could not be the case if the molecule bends
    if(touch(lmp,m,Rstart1,Rend1) || touch(lmp,m,Rstart2,Rend2)) is_touching = true; else is_touching = false;

    if( (d1 > (Length_Molecule-Diameter)) && (d2 > (Length_Molecule -Diameter)) && (is_touching == false) ){
    
      if(dimension == 2) Rstart1[2] = Rend1[2] = Rstart2[2] = Rend2[2] = x[is][2];
 
      if(write_separation) draw_molecule(lmp,m,0);
      sprintf(line,"group father id %d:%d",Bacterium[m].nstart,Bacterium[m].nend); lmp->input->one(line);
      sprintf(line,"delete_bonds father bond ${BondType12} any remove special"); lmp->input->one(line); 
// we assume that a bacterium that splits in two looses all of its eps bonds - this is not a big deal; if there are sourrounding eps molecules, bonds are immediately reformed

      // the splitting bacterium is moved and shortened
      move(lmp, m, Rstart1, Rend1);

      new_m = create_bacterium_and_mol(lmp, Rstart2, Rend2);


      if(write_separation) draw_molecule(lmp,m,1);
      if(write_separation) draw_molecule(lmp,new_m,2);
      // reset the born time
      Bacterium[m].t_born = total_time;
      Bacterium[m].t_reproduction = -t_reproduction*log(Xrandom());
      Bacterium[m].min_reproduction_time = total_time;

      Bacterium[new_m].t_born = total_time;
      Bacterium[new_m].t_reproduction = -t_reproduction*log(Xrandom());
      Bacterium[new_m].min_reproduction_time = total_time;
      // eps production of new molecule
      Bacterium[new_m].t_eps_production = Bacterium[new_m].t_eps_production;
      // torque 
      Bacterium[new_m].Torque = Bacterium[m].Torque;
      Bacterium[new_m].t_mobility = Bacterium[m].t_mobility;


      // both molecules should be in tumbling state after the reproduction, so that they don't get stuck
      Bacterium[m].state = 1;
      Bacterium[m].t_transition = total_time -t_tumble*log(Xrandom());
      Bacterium[new_m].state = 1;
      Bacterium[new_m].t_transition = total_time -t_tumble*log(Xrandom());

      Bacterium[new_m].t_adhesion = total_time -t_adhesion*log(Xrandom());

      if(Xrandom() < 0.3) Bacterium[new_m].frozen = true;

      sprintf(line,"group molecule_new id %d:%d", Bacterium[new_m].nstart, Bacterium[new_m].nend); lmp->input->one(line);
      sprintf(line,"velocity molecule_new zero linear");lmp->input->one(line);
      sprintf(line,"velocity father zero linear");lmp->input->one(line);
      sprintf(line,"group molecule_new delete"); lmp->input->one(line);
      sprintf(line,"group father delete"); lmp->input->one(line);

      inserted++;
      // ENERGY MINIMIZATION - COULD CREATE HUGE OVERLAPS WITH EPS PARTICLES THAT NEED TO BE RELAXED
//      sprintf(line,"group bact type 1"); lmp->input->one(line);
//      sprintf(line,"fix freeze bact setforce 0.0 0.0 0.0"); lmp->input->one(line);
//      sprintf(line,"minimize 1e-1 1.0e-1 100 100"); lmp->input->one(line);
//      sprintf(line,"unfix freeze"); lmp->input->one(line);


      /*
      // rewire the eps
      nEPS = Bacterium[m].num_EPS;
      for(int i = 0; i < nEPS; i++){
	 index_eps_S[i] = Bacterium[m].EPSstart[i];
	 index_eps_E[i] = Bacterium[m].EPSend[i];
      }
      Bacterium[m].num_EPS = 0;
      for(int i = 0; i < nEPS; i++){
        // unwrapped position eps
      	is = lmp->atom->map(index_eps_S[i]); 
        lmp->domain->unmap(x[is],image[is],unwrap_ns);
        min_dist = 100;
        father = true;
	for(int k = 0; k < NBeads_Molecule; k++){
          ie = lmp->atom->map(Bacterium[m].nstart+k); // local id
          lmp->domain->unmap(x[ie],image[ie],unwrap_ne);
          d1 = powl(unwrap_ns[0]-unwrap_ne[0],2) + powl(unwrap_ns[1]-unwrap_ne[1],2) +  powl(unwrap_ns[2]-unwrap_ne[2],2);
	  if(d1 < min_dist){ min_dist = d1; selected = Bacterium[m].nstart+k;}
	}
	for(int k = 0; k < NBeads_Molecule; k++){
          ie = lmp->atom->map(Bacterium[new_m].nstart+k); // local id
          lmp->domain->unmap(x[ie],image[ie],unwrap_ne);
          d1 = powl(unwrap_ns[0]-unwrap_ne[0],2) + powl(unwrap_ns[1]-unwrap_ne[1],2) +  powl(unwrap_ns[2]-unwrap_ne[2],2);
	  if(d1 < min_dist){ min_dist = d1; selected = Bacterium[new_m].nstart+k; father = false;}
	}
        sprintf(line,"create_bonds single/bond %d %d %d",3,index_eps_S[i],selected); lmp->input->one(line);
	if(father) selected = m; else selected = new_m;
        i = Bacterium[selected].num_EPS;
        Bacterium[selected].EPSstart[i] = index_eps_S[i];
        Bacterium[selected].EPSend[i]   = index_eps_E[i];
        Bacterium[selected].num_EPS++;
      }
      // end rewiring
      */

    }else{
      // if we cannot split the molecule, then we try again after some time
      // conversely if we try at every timestep and the simulation slows down too much
      Bacterium[m].min_reproduction_time = total_time+t_reproduction/5;
    }
  }
  sprintf(buffer,"group bodies molecule != 0"); lmp->input->one(buffer);
  sprintf(buffer,"neigh_modify exclude molecule/intra bodies"); lmp->input->one(buffer); // if we exclude this, then no eps-eps bond are formed!

  if(inserted) relax_locally(m, new_m);
/*
  sprintf(buffer,"run 0"); lmp->input->one(buffer);
  cout << "############################ ENERGY AFTER SPLITTING " << get_single_variable(lmp,"PE") << endl;
  if(write_separation) draw_molecule(lmp,m,1);
  if(write_separation) draw_molecule(lmp,new_m,2);
  getchar();
*/
}

void C_Grow::relax_locally(int n, int nn){
//  int inum = lmp->force->pair->list->inum;
  int jnum;
//  int* ilist = lmp->force->pair->list->ilist;
  int* jlist;
  int* numneigh = lmp->force->pair->list->numneigh;
  int** firstneigh = lmp->force->pair->list->firstneigh;
  char buffer[100];
  int m, id1,index;
  sprintf(buffer,"run 0"); lmp->input->one(buffer);
  int added = 0;

  for(int k = 0; k < 2; k++){
    for(int s = 0; s < NBeads_Molecule; s++){
      if(k == 0) id1 = Bacterium[n].nstart+s;
      else id1 = Bacterium[nn].nstart+s;
      index = lmp->atom->map(id1); // internal index of the atom
      jlist = firstneigh[index]; // list of neighbors of the particles
      jnum = numneigh[index]; // number of neighbors of the particle
      for(int jj = 0; jj < jnum; jj++){
        m = jlist[jj]; // local index of particle 
        if(lmp->atom->type[m] == 2){ // only the eps relaxes
	  sprintf(buffer,"group relax id %d", lmp->atom->tag[m]); lmp->input->one(buffer);
          added++;
	}
      }
    }
  }

  if(added > 0){
    sprintf(buffer,"group freeze subtract all relax"); lmp->input->one(buffer);
    sprintf(line,"fix freeze freeze setforce 0.0 0.0 0.0"); lmp->input->one(line);
    close_dump(lmp);
    int timestep = get_single_variable(lmp,"TimeStep");
    sprintf(line,"minimize 1e-1 1.0e-1 100 100"); lmp->input->one(line);
    sprintf(line,"unfix freeze"); lmp->input->one(line);
    sprintf(line,"reset_timestep %d", timestep); lmp->input->one(line);
    open_dump(lmp,step_dumps);
    sprintf(buffer,"group relax delete"); lmp->input->one(buffer);
  }
}

// extract the value of a variable defined in lammps; this allows to define
// all relevant parameters in the file "in.setup"
double get_single_variable(LAMMPS* lmp, const char* pippo){
  char variable[100];
  strcpy(variable, pippo);
  char S_NULL[100]; sprintf(S_NULL,"NULL");
  double* pippone = (double *) lammps_extract_variable(lmp, variable, S_NULL);
  return pippone[0];
}

// execute the lammps script "im.setup", and get some info about the particles
void setup(LAMMPS* lmp, int me){
  for(int m = 0; m < Nmax_bacteria; m++) Bacterium[m].exist = -1;
  lmp->input->file("in.setup");
  Length_Molecule = get_single_variable(lmp,"LL");
  NBeads_Molecule = int(get_single_variable(lmp,"NumAtomMolecule"));
  Diameter = get_single_variable(lmp,"Diameter");
  Lbox = get_single_variable(lmp,"Lbox");
  Frun = get_single_variable(lmp,"Frun");
  Torque = get_single_variable(lmp,"Torque");
  dt_integration = get_single_variable(lmp,"dt_integration");
  viscosity = get_single_variable(lmp,"viscosity");
  Velocity= get_single_variable(lmp,"velocity");
  t_run = get_single_variable(lmp,"t_run");
  t_tumble = get_single_variable(lmp,"t_tumble");
  t_reproduction = get_single_variable(lmp,"t_reproduction");
  t_adhesion = get_single_variable(lmp,"t_adhesion");
  timestep_splitting_check_interval = int( get_single_variable(lmp,"splitting_check_interval")*1.0/dt_integration);
  t_eps = get_single_variable(lmp,"t_eps");
  Temperature = get_single_variable(lmp,"Temperature");
  cutoff_11 = get_single_variable(lmp,"cut11");
  cutoff_12 = get_single_variable(lmp,"cut12");
  cutoff_22 = get_single_variable(lmp,"cut22");
  sigma_11 = get_single_variable(lmp,"sigma11");
  sigma_12 = get_single_variable(lmp,"sigma12");
  sigma_22 = get_single_variable(lmp,"sigma22");
  MassEPS = get_single_variable(lmp,"MassEps");
  step_dumps = get_single_variable(lmp,"timestep_dump");

  bond_type_12 = get_single_variable(lmp,"BondType12");
  bond_type_22 = get_single_variable(lmp,"BondType22");
  
  dimension = get_single_variable(lmp,"dimension");
  num_bond_freeze = get_single_variable(lmp,"NumBondFreeze");
  zoom = get_single_variable(lmp,"zoom");
  Kbond_bacterium = get_single_variable(lmp,"K_bacterium");

  total_simulation_steps = get_single_variable(lmp,"total_simulation_steps");

  for(int n = 0; n < Nmax_bacteria; n++){
    Bacterium[n].exist = -1;
    Bacterium[n].molecule_type = -1;
  }

  if(get_single_variable(lmp,"USE_EPS") == 0) USE_EPS = true; else USE_EPS = false;
}

void create_template_molecule(LAMMPS* lmp, int me, double EE[3]){
  double N = NBeads_Molecule;
  double LL = Length_Molecule; // initial lenght; the final one will be the double
  double length;
  char line[1024];
  system("rm template.bacteria*"); // eliminate existing templates
    sprintf(line,"template.bacteria");
  ofstream outfile(line);
  length = LL;
  outfile << "#template for a bacteria with " << N  << "atoms, of lenght " << length << endl << endl;
  outfile << N << " atoms" << endl;
//  outfile << N-1 << " bonds" << endl;
  outfile << N-2 << " angles" << endl;
  outfile << endl;
  outfile << "Coords" << endl << endl;
  for(int n = 0; n < N; n++){
    outfile << n+1 << "	" << EE[0]*n*1.0/(N-1)-EE[0]/2 << "	" << EE[1]*n*1.0/(N-1)-EE[1]/2 << "	" << EE[2]*n*1.0/(N-1)-EE[2]/2 << endl;
  }
  outfile << endl;
  outfile << "Types" << endl << endl;
  for(int n = 0; n < N; n++) outfile << n+1 << "	" << 1 << endl;
  outfile << endl;
//  outfile << "Bonds" << endl << endl;
//  for(int n = 1; n < N; n++) outfile << n << "	" << 1 << "	" << n << "	" << n+1 << endl;
//  outfile << endl;
  outfile << "Angles" << endl << endl;
  for(int n = 2; n < N; n++) outfile << n-1 << "	" << 1 << "	" << n-1 << "	" << n << "	" << n+1 << endl;
  outfile.close();

  sprintf(line,"molecule TemplateBacteria template.bacteria");
  lmp->input->one(line);
}


// calback function - this is called every timestep
void force_callback(void *ptr, bigint ntimestep, int nlocal, int *id, double **x, double **f) {
//  TimeMeas.set_begin(0); 
  double f0,fx,fy, delta_rel;
  double dr[3], end_to_end, dis, rc[3];

  Info *info = (Info *) ptr;
  for(int q = 0; q < nlocal; q++) f[q][0] = f[q][1] = f[q][2] = 0.0;

  char buffer[1000];

  // set to zero the number of bacteria to be splitted, and the StopRun variable
  Grow.num_to_split = 0;
  double unwrap[50][3];
  int index[50];
  int state;
  imageint *image = info->lmp->atom->image;

  double Ls = Length_Molecule-Diameter;
  double Le = (2*Length_Molecule-Diameter)*1.2;
  double Lsplit = (2*Length_Molecule-Diameter)*1.01;
  double ActualL;

  int timestep = get_single_variable(info->lmp,"TimeStep");

  bool can_stop = false;
  if( (timestep >= timestep_splitting_check_interval) && (timestep*1.0)/timestep_splitting_check_interval == int(timestep*1.0/timestep_splitting_check_interval)) can_stop = true;


// SET TO FALSE NOT ALLOW REPRODUCTION - USED IN CALCULATION OF MSD
//  can_stop = false;

  for(int m = 0; m < num_bacteria; m++){
    // get the unwrapped coordinates of the particles
    for(int n = 0; n < NBeads_Molecule; n++){
      index[n] = info->lmp->atom->map(n+Bacterium[m].nstart); // local id
      info->lmp->domain->unmap(x[index[n]],image[index[n]],unwrap[n]);
    }
    // this is the end to the end distance the bacterium would like to have give its age
    end_to_end = Ls+(Le-Ls)*(total_time-Bacterium[m].t_born)*1.0/Bacterium[m].t_reproduction;
    if(end_to_end > Le) end_to_end = Le;
    // this is the corresponding length of all of the bonds 
//    eq_bond_length = end_to_end/(NBeads_Molecule-1);

    // loop over all bonds of a bacterium; we add a force promoting the desired bond length
    // the force is applied to two beads sharing the bond

    // extensive force applied at the two ends
    dis = 0;
    for(int q = 0; q < dimension; q++){
      dr[q] = unwrap[NBeads_Molecule-1][q]-unwrap[0][q];
      if(dr[q] > 0.5*Lbox) dr[q] = Lbox-dr[q]; else if(dr[q] < -0.5*Lbox) dr[q] += Lbox;
      dis += dr[q]*dr[q];
    }
    dis = sqrt(dis);
    ActualL = dis;
    delta_rel = (dis-end_to_end);
    f0 = Kbond_bacterium*( delta_rel*(1+10*delta_rel*delta_rel) ); // harmonic + quartic
    for(int q = 0; q < dimension; q++){
       f[index[NBeads_Molecule-1]][q] += -f0*dr[q]/dis; 
       f[index[0]][q] += +f0*dr[q]/dis; 
    };

    // force applied at all the other beads, to make the beads equally spaced 
    for(int n = 1; n <= NBeads_Molecule-2; n++){
      dis = 0;
      for(int q = 0; q < dimension; q++){
        dr[q] = unwrap[0][q]+(unwrap[ NBeads_Molecule-1 ][q]-unwrap[ 0 ][q])*n*1.0/(NBeads_Molecule-1); // ideal position
        dr[q] = dr[q]-unwrap[n][q]; // distance from ideal position
        dis += dr[q]*dr[q];
      }
      dis = sqrt(dis);
      f0 = 100*Kbond_bacterium*( dis*(1+10*dis*dis) ); // harmonic + quartic
      for(int q = 0; q < dimension; q++){
        if(dis > 0) f[index[n]][q] += f0*dr[q]/dis; 
      }
    }

    
    if(Bacterium[m].t_adhesion < total_time) Bacterium[m].frozen = true; 

    /*
    if( Bacterium[m].frozen == false){
      // get info about the versor and the length of the molecule
      ns = Bacterium[m].nstart; // gloab id
      ne = Bacterium[m].nend;   // global id
      is = info->lmp->atom->map(ns); // local id
      ie = info->lmp->atom->map(ne);   // local id
      norm2 = 0;
      // calculate the end-tp-end distance; norm is use to add the torque
      for(int k = 0; k < dimension; k++){
        dr[k] = x[ie][k]-x[is][k]; 
        if(dr[k] > 0.5*Lbox) dr[k] -= Lbox; 
        if(dr[k] < -0.5*Lbox) dr[k] += Lbox;
        norm2 += dr[k]*dr[k];
      }
      norm = sqrt(norm2);
    }       
    */

    // reproduction
    if( can_stop && (ActualL > Lsplit) && (Bacterium[m].min_reproduction_time < total_time)){
      sprintf(buffer,"variable StopRun equal %d", timestep); info->lmp->input->one(buffer); 
      sprintf(buffer,"variable pippo equal %d", m); info->lmp->input->one(buffer); 
      Grow.tosplit[Grow.num_to_split] = m;
      Grow.num_to_split++;
    }

    // eps production
    if(USE_EPS && Bacterium[m].frozen){ 
      Bacterium[m].produce_eps = false;
      if(can_stop && (total_time > Bacterium[m].t_eps_production)){
         Bacterium[m].produce_eps = true;
         // we set the reproduction time; we want the production of different bacteria to be somehow in sync, as otherwise we call to frequently the eps production function
         Bacterium[m].t_eps_production = t_eps*int( (total_time+t_eps)/t_eps);
         sprintf(buffer,"variable StopRun equal 2"); info->lmp->input->one(buffer);
         sprintf(buffer,"variable bacteria equal %d", m); info->lmp->input->one(buffer); 
         sprintf(buffer,"variable tempo equal %d", timestep); info->lmp->input->one(buffer); 
      }
    }

    // motility
    if(Bacterium[m].frozen == false){
      dis = 0;
      for(int q = 0; q < dimension; q++){
        dr[q] = unwrap[NBeads_Molecule-1][q]-unwrap[0][q];
        dis += dr[q]*dr[q];
      }
      dis = sqrt(dis);
      if(Bacterium[m].molecule_type ==1){
        state = Bacterium[m].state;
        if(state == 0){ // run
          // add a force to all atoms making the bacterium
          for(int i = 0; i < NBeads_Molecule; i++){
            f[index[i]][0] += Bacterium[m].Force*dr[0]/dis;
            f[index[i]][1] += Bacterium[m].Force*dr[1]/dis;
          }
        }
        // tumble
	if(state == 1){ // tumble
          // add a force to all atoms making the bacterium, so that the angular velocity does not depend on the particle size
	  // center of the molecule
      	  rc[0] = (unwrap[0][0]+unwrap[(NBeads_Molecule-1)][0])/2;
          rc[1] = (unwrap[0][1]+unwrap[(NBeads_Molecule-1)][1])/2;
	  for(int i=0;i<NBeads_Molecule-1;i++){
            // distance from the center
      	    dr[0] = unwrap[i][0]-rc[0];
            dr[1] = unwrap[i][1]-rc[1];
	    fx = Bacterium[m].Torque*dr[1];
	    fy = -Bacterium[m].Torque*dr[0];
	    f[index[i]][0] += fx;
            f[index[i]][1] += fy;
          }
        }

	// change the mobility state if needed

        if(  (state == 0) || (state == 1) ){ // if mobile
          if(total_time > Bacterium[m].t_transition){ // need to change state from run to tumble
            if(Bacterium[m].state == 0){ // from running to tumbling
              Bacterium[m].state = 1;
              Bacterium[m].t_transition = total_time -t_tumble*log(Xrandom());
              // the next force can point in either of the two direction
              Bacterium[m].Force = Frun;
            }else if(Bacterium[m].state == 1){ // from tumbling to running
              Bacterium[m].state = 0;
              Bacterium[m].t_transition = total_time-t_run*log(Xrandom());
              // the next torque has a random sign
              if(Xrandom() < 0.5) Bacterium[m].Torque = Torque;
              else Bacterium[m].Torque = -Torque;
            }
          }
        };
      };
    }
  }
//  TimeMeas.add_time(info->lmp,0);

/*
  // we add the bonds
  TimeMeas.set_begin(2); 
  int timestep = get_single_variable(info->lmp,"TimeStep");
  int timestep_bond = get_single_variable(info->lmp,"timestep_bond");
  if( (timestep/timestep_bond) == (timestep*1.0/timestep_bond) ){
    BondClass.setup(info->lmp);
    BondClass.add_bonds();
  }
  TimeMeas.add_time(info->lmp,2);
//  outfile_f0 << total_time << "	" << max_f0 << endl;
*/
  total_time = total_time+dt_integration;
}






void fix_times(int m){
  if(Bacterium[m].exist > 0){
    if(Xrandom() < 0.5) Bacterium[m].Force = Frun;
    else Bacterium[m].Force = -Frun;
    if(Xrandom() < 0.5) Bacterium[m].Torque = Torque;
    else Bacterium[m].Torque = -Torque;
    if(Bacterium[m].state == 0){ // the bacterium is running
       Bacterium[m].t_transition = total_time - t_run*log(Xrandom());
    };
    if(Bacterium[m].state == 1){
       Bacterium[m].t_transition = total_time - t_tumble*log(Xrandom());
    };
  }
//  Bacterium[m].t_eps_production = total_time - t_eps*log(Xrandom()); // time at which eps is produced
  Bacterium[m].t_eps_production = total_time + t_eps; // time at which eps is produced
  Bacterium[m].t_reproduction = -t_reproduction*log(Xrandom()); 
  Bacterium[m].t_adhesion = -t_adhesion*log(Xrandom()); 
  Bacterium[m].min_reproduction_time = 0; // reproduce at least after this time
  Bacterium[m].t_born = total_time;
  Bacterium[m].produce_eps = false;

}


void run_and_tumble(){
  for(int m=0; m< Nmax_bacteria; m++) fix_times(m);
}

void produce_eps(LAMMPS* lmp){
  double** x;
  imageint *image;
  double R[3];
  int id, n, index, id1, index1;
  char line[1024];
  int natoms;
  double dist_insertion = cutoff_12*1.02;
  double theta, phi, dx, dy, dz, dis2;
  bool loop, touch_bacterium;
  double cutsq[3];
  double unwrap[3], unwrap1[3];
  cutsq[1] = powl(cutoff_12,2);
  cutsq[2] = powl(cutoff_22,2);
  sprintf(line,"group g_bact type 1"); lmp->input->one(line);  // bacterial particles - they do not change during eps insertion
  sprintf(line,"group g_eps type 2"); lmp->input->one(line); // eps particles 

  //let's try to access the neighbor list
  int jnum;
  int* jlist;
  int** firstneigh = lmp->force->pair->list->firstneigh;
  int* numneigh = lmp->force->pair->list->numneigh;

  for(int m = 0; m < num_bacteria; m++){
    if( (Bacterium[m].exist > 0) && (Bacterium[m].produce_eps == true)){
      Bacterium[m].produce_eps = false;
      do{
        id = Bacterium[m].nstart+int(Xrandom()*NBeads_Molecule); // global id of a bacterium atom
        dist_insertion = 0.5*(sigma_11+sigma_22);
        index = lmp->atom->map(id); // local id
        x = lmp->atom->x; // we do this as x might change when inserting a new atom
        image = lmp->atom->image;
        lmp->domain->unmap(x[index],image[index],unwrap);
        // we select a random point at a given distance from the atom
        if(dimension == 2){
          theta = Xrandom()*2*M_PI;
          R[0] = unwrap[0]+dist_insertion*cos(theta); 
          R[1] = unwrap[1]+dist_insertion*sin(theta);
          R[2] = unwrap[2];
        }else{
          theta = Xrandom()*2*M_PI;
          phi = Xrandom()*2*M_PI;
          R[0] = dist_insertion*sin(theta)*cos(phi); 
          R[1] = dist_insertion*sin(theta)*sin(phi); 
          R[2] = dist_insertion*cos(phi); 
          R[0] += unwrap[0]; R[1] += unwrap[1]; R[2] += unwrap[2];
        }
	touch_bacterium = false;
        for(int s = 0; s < NBeads_Molecule; s++){
          id1 = Bacterium[m].nstart+s;
          index1 = lmp->atom->map(id1); // local id
          lmp->domain->unmap(x[index1],image[index1],unwrap1);
          dx = fabs(R[0]-unwrap1[0]); dx = dx-Lbox*int(dx/Lbox);
          dy = fabs(R[1]-unwrap1[1]); dy = dy-Lbox*int(dy/Lbox);
          dz = fabs(R[2]-unwrap1[2]); dz = dz-Lbox*int(dz/Lbox);
          dis2 = dx*dx + dy*dy + dz*dz;
	  if(dis2 < dist_insertion*dist_insertion) touch_bacterium = true;
	}
      }while(touch_bacterium);
      // insertion cheking for overlaps using groups - this is much faster than introducing the particle and check for overlap using lammps, as commented below
      loop = true;
      // overlap with the bottom plane
      if((dimension == 3) && (R[2] < cutoff_22)){ loop = false;}
   
      // check for overlap
      for(int s = 0; s < NBeads_Molecule; s++){
        id1 = Bacterium[m].nstart+s;
        index = lmp->atom->map(id1); // internal index of the atom
        jlist = firstneigh[index]; // list of neighbors of the particles
        jnum = numneigh[index]; // number of neighbors of the particle
        for(int jj = 0; jj < jnum; jj++){
          n = jlist[jj]; // local index of particle 
          dx = fabs(x[n][0]-R[0]); dx = dx - Lbox*int(dx/Lbox); 
          dy = fabs(x[n][1]-R[1]); dy = dy - Lbox*int(dy/Lbox); 
          dz = fabs(x[n][2]-R[2]); dz = dz - Lbox-int(dz/Lbox); 
          dis2 = dx*dx+dy*dy+dz*dz;
          if(dis2 < cutsq[lmp->atom->type[n]]){ loop = false; jj = jnum;}
	}
      }
      natoms = static_cast<int> (lmp->atom->natoms);
      if(loop){
        sprintf(line,"create_atoms 2 single %lf %lf %lf", 0.5*Lbox, 0.5*Lbox, 0.5*Lbox); lmp->input->one(line); 
        sprintf(line,"group atomo id %d", natoms+1); lmp->input->one(line);
        sprintf(line,"displace_atoms atomo move %lf %lf %lf", R[0]-0.5*Lbox, R[1]-0.5*Lbox, R[2]-0.5*Lbox); lmp->input->one(line);
	sprintf(line,"group atomo delete"); lmp->input->one(line);
        numEPS++;
      }
    }
  }
  if(numEPS > 0){
    sprintf(line,"fix freeze g_bact setforce 0.0 0.0 0.0"); lmp->input->one(line);
    close_dump(lmp);
    int timestep = get_single_variable(lmp,"TimeStep");
    sprintf(line,"minimize 1e-1 1.0e-1 100 100"); lmp->input->one(line);
    sprintf(line,"unfix freeze"); lmp->input->one(line);
    sprintf(line,"reset_timestep %d", timestep); lmp->input->one(line);
    open_dump(lmp,step_dumps);
  }

  sprintf(line,"group g_bact delete"); lmp->input->one(line);
}

/*
void produce_eps(LAMMPS* lmp){
  double** x;
  imageint *image;
  double R[3];
  int id, n, index, id1, index1;
  int index_eps_start;
  char line[1024];
  int natoms;
  double dist_insertion = cutoff_12*1.02;
  double theta, phi, dx, dy, dz, dis2;
  bool loop, touch_bacterium;
  double cutsq[3];
  double unwrap[3], unwrap1[3];
  cutsq[1] = powl(cutoff_12,2);
  cutsq[2] = powl(cutoff_22,2);
  sprintf(line,"group g_bact type 1"); lmp->input->one(line);  // bacterial particles - they do not change during eps insertion
  sprintf(line,"group g_eps type 2"); lmp->input->one(line); // eps particles 
  for(int m = 0; m < num_bacteria; m++){
    if( (Bacterium[m].exist > 0) && (Bacterium[m].produce_eps == true) && (Bacterium[m].frozen == false)){
      Bacterium[m].produce_eps = false;

      do{
        if(Bacterium[m].num_EPS < MaxEpsHarms){
          id = Bacterium[m].nstart+int(Xrandom()*NBeads_Molecule); // global id of a bacterium atom
          dist_insertion = cutoff_12*1.02;
        }else{
          index_eps_start = int(Xrandom()*Bacterium[m].num_EPS);
     	  id = Bacterium[m].EPSend[index_eps_start];
          dist_insertion = cutoff_22*1.02;
//        cout << "si riproduce eps" << id << " of " << lmp->atom->natoms << endl; getchar();
        }
//        cout << "ID of production " << id << endl;
        index = lmp->atom->map(id); // local id
        // we select a random point at a given distance from the atom
        x = lmp->atom->x; // we do this as x might change when inserting a new atom
        image = lmp->atom->image;
        lmp->domain->unmap(x[index],image[index],unwrap);
        if(dimension == 2){
          theta = Xrandom()*2*M_PI;
          R[0] = unwrap[0]+dist_insertion*cos(theta); 
          R[1] = unwrap[1]+dist_insertion*sin(theta);
          R[2] = unwrap[2];
        }else{
          theta = Xrandom()*2*M_PI;
          phi = Xrandom()*2*M_PI;
          R[0] = dist_insertion*sin(theta)*cos(phi); 
          R[1] = dist_insertion*sin(theta)*sin(phi); 
          R[2] = dist_insertion*cos(phi); 
          R[0] += unwrap[0]; R[1] += unwrap[1]; R[2] += unwrap[2];
        }
	touch_bacterium = false;
        for(int s = 0; s < NBeads_Molecule; s++){
          id1 = Bacterium[m].nstart+s;
          index1 = lmp->atom->map(id1); // local id
          lmp->domain->unmap(x[index1],image[index1],unwrap1);
          dis2 = powl(R[0]-unwrap1[0],2) + powl(R[1]-unwrap1[1],2) + powl(R[2]-unwrap1[2],2);

//	  cout << s << "	" << dis2 << "	" << cutsq[1] << "	" << cutsq[2] << "	" << id1 << "	" << id << "	" << theta << "	" << dist_insertion*dist_insertion << endl; 

	  if(dis2 < cutsq[1]) touch_bacterium = true;
	}
//        cout << "AAA " << endl; getchar();
      }while(touch_bacterium);
      // insertion cheking for overlaps using groups - this is much faster than introducing the particle and check for overlap using lammps, as commented below
      loop = true;
      // overlap with the bottom plane
      if((dimension == 3) && (R[2] < cutoff_22)){ loop = false;}
   
        // check for overlap
      natoms = static_cast<int> (lmp->atom->natoms);
      n = -1;
      while( loop && (n < natoms-1)){
        n++;
        dx = fabs(x[n][0]-R[0]); if(dx > Lbox*0.5) dx = Lbox-dx; if(dx > cutsq[1]){ continue;}
        dy = fabs(x[n][1]-R[1]); if(dy > Lbox*0.5) dy = Lbox-dy; if(dy > cutsq[1]){ continue;}
        dz = fabs(x[n][2]-R[2]); if(dz > Lbox*0.5) dz = Lbox-dz; if(dz > cutsq[1]){ continue;}
        dis2 = dx*dx+dy*dy+dz*dz;
        if(dis2 < cutsq[lmp->atom->type[n]]) loop = false;
      }
      if(loop){
        sprintf(line,"create_atoms 2 single %lf %lf %lf", 0.5*Lbox, 0.5*Lbox, 0.5*Lbox); lmp->input->one(line); 
        sprintf(line,"group atomo id %d", natoms+1); lmp->input->one(line);
        sprintf(line,"displace_atoms atomo move %lf %lf %lf", R[0]-0.5*Lbox, R[1]-0.5*Lbox, R[2]-0.5*Lbox); lmp->input->one(line);
	sprintf(line,"group EPSfirst id %d",natoms+1); lmp->input->one(line);
	sprintf(line,"group atomo delete"); lmp->input->one(line);
        if(Bacterium[m].num_EPS < MaxEpsHarms){
	  Bacterium[m].EPSstart[Bacterium[m].num_EPS] = natoms+1;
	  Bacterium[m].EPSend[Bacterium[m].num_EPS] = natoms+1;
	  Bacterium[m].num_EPS++;
          sprintf(line,"create_bonds single/bond %d %d %d",2,id,natoms+1); lmp->input->one(line);
	}else{
          sprintf(line,"create_bonds single/bond %d %d %d",3,id,natoms+1); lmp->input->one(line);
	  Bacterium[m].EPSend[index_eps_start] = natoms+1;
	}
        numEPS++;
	// create bond
      }
    }
  }
  sprintf(line,"group g_bact delete"); lmp->input->one(line);
}
*/
