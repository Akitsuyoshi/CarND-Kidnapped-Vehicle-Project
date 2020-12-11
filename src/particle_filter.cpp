/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;
using std::default_random_engine;
using std::normal_distribution;
using std::numeric_limits;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  default_random_engine gen;
  double std_x = std[0], std_y = std[1], std_theta = std[2];

  // For normal ditribution of x, y, and theta respectively
  normal_distribution<double> dist_x(x, std_x);
  normal_distribution<double> dist_y(y, std_y);
  normal_distribution<double> dist_theta(theta, std_theta);

  num_particles = 2000;  // TODO: Set the number of particles
  for (unsigned int i = 0; i < num_particles; i++) {
    Particle p = {i, dist_x(gen), dist_y(gen), dist_theta(gen), 1};
    particles.push_back(p);
    weights.push_back(1);
  }
  // Finish initialization
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  default_random_engine gen;
  double std_x = std_pos[0], std_y = std_pos[1], std_theta = std_pos[2];

  for(Particle& p: particles) {
    double theta = p.theta * (yaw_rate * delta_t);
    double x = p.x + velocity / yaw_rate * (sin(theta)- sin(p.theta));
    double y = p.y + velocity / yaw_rate * (cos(p.theta) - cos(theta));

    // For normal ditribution of x, y, and theta respectively
    normal_distribution<double> dist_x(x, std_x);
    normal_distribution<double> dist_y(y, std_y);
    normal_distribution<double> dist_theta(theta, std_theta);

    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  for(LandmarkObs obs: observations) {
    // Get x and y observation codinates
    double x_obs = obs.x;
    double y_obs = obs.y;

    // Placeholder for holding min distance between predicted and observed measurement
    double distance_min = numeric_limits<double>::infinity();

    // Find the most colosest predicted measurement
    for(LandmarkObs predicted_obs: predicted) {
      double current_distance = dist(obs.x, obs.y, predicted_obs.x, predicted_obs.y);
      if (distance_min > current_distance) {
        continue;
      }

      // Update observation with current most closest predicted measurement
      obs.id = predicted_obs.id;
      obs.x = predicted_obs.x;
      obs.y = predicted_obs.y;
      // Set current distance as a distance min
      distance_min = current_distance;
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  double std_x = std_landmark[0], std_y = std_landmark[1];
  double gauss_norm = 1 / (2 * M_PI * std_x * std_y);

  for(Particle& p: particles) {
    // Get x and y partial codinates
    double x_part = p.x;
    double y_part = p.y;

    vector<LandmarkObs> transformed_observations = observations;
    vector<LandmarkObs> nearest_observations;
    vector<LandmarkObs> predicted;
    vector<int> associations;
    vector<double> sense_x;
    vector<double> sense_y;

    // Update each observation cordinates from vehicles to map system
    for(LandmarkObs obs: transformed_observations) {
      // Get x and y observation cordinates, on a vehicle's system
      double x_obs = obs.x;
      double y_obs = obs.y;
      const double theta = -M_PI / 2;

      // Get observation cordinates, on a map system
      double x_map_obs = x_part + (cos(theta) * x_obs) - (sin(theta) * y_obs);
      double y_map_obs = y_part + (sin(theta) * x_obs) + (cos(theta) * y_obs);

      // Update cordinates
      obs.x = x_map_obs;
      obs.y = y_map_obs;
    }

    // Assign observations having map cordinate to nearest
    nearest_observations = transformed_observations;

    // Get predicted landmarks from Map
    for(Map::single_landmark_s landmark: map_landmarks.landmark_list) {
      double distance = dist(x_part, y_part, landmark.x_f, landmark.y_f);
      if (distance > sensor_range) {
        continue;
      }
      predicted.push_back(LandmarkObs{landmark.id_i, landmark.x_f, landmark.y_f});
    }

    // Update observations with nearest Map landmarks
    dataAssociation(predicted, nearest_observations);

    // Calculate weights
    for (unsigned int i = 0; i < nearest_observations.size(); i++) {
      LandmarkObs transformed_obs = transformed_observations[i];
      LandmarkObs nearest_obs = nearest_observations[i];

      double exponent = (pow(transformed_obs.x - nearest_obs.x, 2) / (2 * pow(std_x, 2)))
                      + (pow(transformed_obs.y - nearest_obs.y, 2) / (2 * pow(std_y, 2)));

      // Update weight
      p.weight *= gauss_norm * exp(-exponent);

      // Update associations, sense_x, and sense_y
      associations.push_back(nearest_obs.id);
      sense_x.push_back(nearest_obs.x);
      sense_y.push_back(nearest_obs.y);
    }

    // Set associations to partial
    SetAssociations(p, associations, sense_x, sense_y);
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */

}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}